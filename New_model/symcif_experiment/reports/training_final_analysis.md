# SymCIF Experiment Final Training Analysis

Date: 2026-05-19

## Completion Status

All three training runs completed:

| experiment | config | log | status |
| --- | --- | --- | --- |
| baseline | `configs/exp_baseline.yaml` | `runs/logs/exp_baseline.log` | completed |
| cf_like | `configs/exp_cf_like.yaml` | `runs/logs/exp_cf_like.log` | completed |
| symcif_v1 | `configs/exp_symcif_v1.yaml` | `runs/logs/exp_symcif_v1.log` | completed |

The final GPU check showed both GPUs idle after training.

## Corpus Size

| mode | vocab | train tokens | val tokens | train token reduction vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 405 | 2055575 | 202345 | 0.0% |
| cf_like | 405 | 1253082 | 123176 | 39.0% |
| symcif_v1 | 405 | 1421459 | 139673 | 30.8% |

`symcif_v1` uses 13.4% more train tokens than `cf_like`, but still uses 30.8% fewer tokens than the full CIF baseline.

## Validation Loss Summary

| experiment | best step | best train loss | best val loss | final step | final train loss | final val loss | final-best val gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 750 | 0.3609 | 0.3845 | 3500 | 0.0262 | 1.1752 | 0.7907 |
| cf_like | 750 | 0.5635 | 0.6113 | 3500 | 0.0273 | 2.0328 | 1.4215 |
| symcif_v1 | 750 | 0.5041 | 0.5390 | 3500 | 0.0273 | 1.7485 | 1.2095 |

All three experiments reached their best validation loss at step 750. After that, train loss continued to decrease while validation loss rose sharply, so the 3500-step setting is too long for this 5000-sample training split.

## Main Observations

1. Training is stable: no NaN, CUDA crash, or stalled process was observed after the initial device-ordinal launch issue was fixed.
2. The strongest training-loss/validation-loss generalization point is around step 750 for all three formats.
3. `baseline` has the lowest validation loss, but this is not a direct proof that it will generate better structures because the three target formats have different token distributions and different semantic burdens.
4. `symcif_v1` improves over `cf_like` at the best validation point: 0.5390 vs 0.6113, an 11.8% relative reduction in validation loss, while adding 13.4% tokens over `cf_like`.
5. `cf_like` is the most token-efficient format, but it overfits the hardest in this setup and has the worst final validation loss.

## Checkpoint Notes

| experiment | available checkpoints | caveat |
| --- | --- | --- |
| baseline | `runs/exp_baseline/ckpt.pt` | saved at final step 3500, not best step 750 |
| cf_like | `runs/exp_cf_like/ckpt.pt` | saved at final step 3500, not best step 750 |
| symcif_v1 | `runs/exp_symcif_v1/ckpt_best.pt`, `ckpt_last.pt`, `ckpt.pt` | best checkpoint is preserved from step 750 |

The checkpoint logic was fixed before `symcif_v1` started, so `symcif_v1` has both best and last checkpoints. `baseline` and `cf_like` were already running with the older checkpoint behavior, so their best-step checkpoints were overwritten by later overfit checkpoints.

## Recommendation

For the next controlled generation/evaluation stage:

- do not use final-step `baseline` or `cf_like` checkpoints as the main comparison if generalization matters;
- rerun `baseline` and `cf_like` with the fixed checkpoint script or set `max_iters: 750` / early stopping;
- keep `symcif_v1` best checkpoint from `runs/exp_symcif_v1/ckpt_best.pt`;
- for future training configs, use `max_iters` around 750-1000, keep best checkpoint, and consider stronger regularization only after early stopping is in place;
- compare models primarily by generated-structure metrics: parse success, formula consistency, space-group consistency, Wyckoff/multiplicity consistency, structure validity, match rate, and RMSD.

Training loss alone is insufficient to choose the best representation. The current training-only result suggests `symcif_v1` is better than `cf_like`, while baseline remains a strong reference. The next decisive test is generation quality from best checkpoints under identical sampling settings.
