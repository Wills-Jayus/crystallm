# Dead End Registry

Permanently stopped as primary opentry_5 routes:

- source-prior-only tuning
- free-param bank copy
- bundle copy
- pair-delta signature search
- independent source-free prior
- direct MSE-only residual geometry
- post-render coordinate surgery
- pairfield small scans as an inference module
- selector/reranker/energy ranking/anchor-safe insertion
- plain MiniCFJoint fixed-order scale-up without rows>=7 curriculum or objective change: E8011/E8012 increased train/eval scale but rows>=7 match@20 stayed 0 on both grouped folds
- weighted CE-only MiniCFJoint geometry/lattice emphasis: E8015/E8016 improved fold_a overall match but did not replicate rows>=7 match on fold_b, so this is not sufficient without an actual geometry/free-param objective or architecture change
- shallow heteroscedastic geometry sampler alone: E8017/E8018 improved top20 on both folds but did not fix fold_b rows>=7 match and lowered top1/top5, so this setting is not sufficient without stronger symbolic W/A recovery or a larger architecture change
- balanced symbolic+geometry sampler without collision-aware training: E8019/E8020 achieved target-level top20 on both folds but rows>=7 match@20 stayed 0 on both folds; do not promote this setting to dev_gate/val512 without a complex-row collision/structure objective
- representative row-pair auxiliary loss on top of balanced sampler: E8021 fold_b kept match@20 above target but match@5 regressed to 32.81% and rows>=7 match@20 stayed 0; stop this auxiliary as a primary route and move to expanded-structure-aware geometry training
- expanded atom-cloud pair-distance auxiliary loss on top of balanced sampler: E8022 fold_b kept match@20 above target and improved complex-row W/A/skeleton hits, but match@1/5 regressed to 12.50%/25.00% and rows>=7 match@20 stayed 0; do not use this auxiliary alone as the primary route
- cartesian atom-cloud pair-distance auxiliary alone: E8023 fold_b improved overall match@20 to 59.38% but rows>=7 W/A@20 and match@20 stayed 0; do not promote cartesian-only auxiliary to dev_gate/val512
- combined high-complex curriculum plus expanded+cartesian atom-cloud auxiliary at E8024 weights: E8024 fold_b reached 31.25% / 40.62% / 57.81% and restored rows>=7 W/A/skeleton hits, but rows>=7 match@20 stayed 0 with geometry invalidity; do not promote this exact setting to dev_gate/val512 without a stronger continuous-geometry fix
- stronger cartesian weighting with lower sample scale on E8024-style curriculum: E8025 fold_b regressed to 18.75% / 31.25% / 54.69%, rows>=7 W/A@20 collapsed to 0, and collision-like rate increased to 49.06%; stop increasing cartesian weights as the primary fix
- previous-representative-coordinate conditioning alone: E8026 fold_b retained rows>=7 W/A/skeleton@20 at 66.67% but rows>=7 match@20 stayed 0 and top1/top5 regressed from E8024; do not rely on decoder context alone without a stronger validity objective
- exact E8027 local-neighbor cartesian setting as a terminal route: E8027 fold_b exceeded top5/top20 thresholds but rows>=7 match@20 stayed 0 and fold_a is not yet tested; do not promote to dev_gate/val512 without paired-fold consistency and rows>=7 improvement
- exact E8027/E8028 local-neighbor cartesian setting as a terminal paired route: E8028 fold_a produced one rows>=7 positive but did not exceed two aggregate thresholds, while E8027 fold_b exceeded aggregate thresholds but rows>=7 stayed 0; continue tuning rather than promoting this exact setting
- exact active-only local separation setting from E8029 as a terminal route: E8029 fold_b improved aggregate top20 to 60.94% and kept top5 above threshold, but rows>=7 W/A, skeleton, and match@20 all collapsed to 0; do not promote this exact setting or its active-only separation weight/min-separation to dev_gate/val512 without restoring complex symbolic recovery
- simple complex symbolic loss weighting plus active-only local separation: E8030 fold_b restored only rows>=7 skeleton@20 = 33.33%, left rows>=7 W/A/match@20 at 0, and regressed top1/top5 to 21.88%/31.25%; do not keep increasing symbolic CE weights as the primary fix

Allowed reuse: historical train failures may be used only as corruption data for generative training.
