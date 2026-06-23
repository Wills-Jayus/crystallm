# Resume State

Updated: 2026-06-19T14:45:00Z

Current phase: Phase 0-4 data foundation completed; Model B denoiser smoke E8003 completed on GPU; MiniCFJoint-v2 grouped fold_a/fold_b gate0 smokes E8004/E8006 passed; MiniCFJoint small-overfit E8005 was interrupted in the evaluator tail; Model B CIF recovery smokes E8007/E8008 passed on fold_a/fold_b as GT-W/A/free-param diagnostics; Model D fixed-order non-oracle smokes E8009-E8038 completed.

Recent checkpoint: `checkpoints/model_b_denoiser_smoke/best.pt`, `checkpoints/model_b_denoiser_smoke/last.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8019_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8020_fold_a/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8021_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8022_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8023_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8024_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8025_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8026_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8027_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8028_fold_a/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8029_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8030_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8031_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8032_fold_b/eval_only_source.json`, `checkpoints/fixed_order_geom_sampler_smoke_E8033_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8034_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8035_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8036_fold_b/eval_only_source.json`, `checkpoints/fixed_order_geom_sampler_smoke_E8037_fold_b/best_train.pt`, `checkpoints/fixed_order_geom_sampler_smoke_E8038_fold_b/best_train.pt`, and incomplete `checkpoints/minicfjoint_v2_fold_a_gate0_smoke/small_overfit/ckpt_best.pt`. Base checkpoint is copied at `checkpoints/base_crystallm_v1_small/ckpt.pt`.

Current best non-oracle full-CIF direction: E8034/E8036, local-neighbor cartesian geometry plus weak target-relative separation and complex symbolic weighting. E8034 reached match@1/5/20 = 34.38% / 48.44% / 51.56%, exceeding all three aggregate +5pp thresholds on fold_b, but rows>=7 match@20 remains 0. E8036 loaded E8034 eval-only with lower sampling and preserved 34.38% / 48.44% / 51.56%, but rows>=7 match@20 still remained 0. E8037/E8038 tested complex geometry NLL weights 1.5 and 1.2; both regressed early aggregate accuracy, and E8038 lost rows>=7 W/A@20 entirely. E8035 increased target-relative separation to 1.5 and reached 28.12% / 40.62% / 56.25%; only top20 cleared the aggregate target. Rows>=7 diagnostics show `mp-1025577` can produce W/A+skeleton candidates under E8036 but remains invalid, while geometry reweighting tends to lose W/A. The route is not consistent enough for dev_gate or val512. E8007/E8008 GT-W/A/free-param CIF recovery remains the best diagnostic result but is oracle-conditioned and not terminal.

Completed tasks:

- workspace directories;
- grouped split;
- canonical train/dev dataset;
- OOF W/A condition-gap files;
- failure-aware corruption files;
- no-ranking audit and experiment books.
- Model B lattice-only smoke training/eval with checkpoint save/resume.
- MiniCFJoint-v2 grouped fold_a and fold_b gate0 target round-trip smokes.
- Model B fixed-order CIF recovery smoke on fold_a and fold_b.
- Model D fixed-order non-oracle MiniCFJoint path-check, scale-up, rows>=7 curriculum, geometry-heavy CE, heteroscedastic geometry sampler, balanced symbolic+geometry sampler smokes on fold_a/fold_b, fold_b representative row-pair auxiliary smoke, fold_b expanded atom-cloud auxiliary smoke, fold_b cartesian atom-cloud auxiliary smoke, fold_b high-complex combined expanded+cartesian auxiliary smoke, fold_b low-sample-scale stronger-cartesian smoke, fold_b previous-coordinate-conditioned decoder smoke, fold_b local-neighbor cartesian auxiliary smoke, paired fold_a local-neighbor cartesian auxiliary smoke, fold_b active-only local separation smoke, fold_b complex-symbolic weighted active-local smoke, fold_b high local-pair/no-active-separation smoke, fold_b eval-only lower-sampling smoke, fold_b target-relative cartesian separation smoke, fold_b weak-relative plus complex-symbolic smoke, fold_b stronger-relative plus complex-symbolic smoke, fold_b E8034 eval-only lower-sampling smoke, and fold_b complex-geometry-weight smokes.

Unfinished tasks:

- redesign Model D objective/architecture to retain complex-row W/A while calibrating continuous free parameters and cell;
- train Model A/B/C as needed if Model D stalls;
- dev_gate;
- at most one val512 per passed model family;
- freeze best model only if terminal metrics are met.

Next task: use a weaker/targeted geometry intervention or a post-generation deterministic coordinate repair that does not reorder candidates; do not run dev_gate or val512 until paired grouped-fold smokes are directionally consistent and rows>=7 failure is addressed.

Concrete reason not complete: the best non-oracle full CIF generator still produced only one rows>=7 positive on fold_a and none on fold_b, and no model has exceeded the frozen no-ranking +5pp target on grouped folds/dev_gate/val512.
