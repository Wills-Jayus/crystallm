# Hypothesis Ledger

## E8000
- Bottleneck hypothesis: prior evaluations mixed scopes and lacked a prototype-aware train/dev boundary.
- Changed axis: split/canonical representation.
- Why positives could appear: model training should see cleaner labels and no prototype leakage.
- Difference vs failed methods: no selector/reranker/source-prior insertion.
- Expected metric: rows>=7 positive-any and W/A-hit match-fail on grouped folds.
- Failure route: if round-trip fails, repair representation before any model training.

## E8001
- Bottleneck hypothesis: GT W/A teacher forcing creates a train/inference condition gap.
- Changed axis: OOF/predicted/corrupted W/A conditioning.
- Why positives could appear: generator learns to recover from realistic W/A errors.
- Expected metric: W/A-hit match-fail down on both grouped folds.
- Failure route: if OOF W/A is too weak, move to joint W/A+geometry model.

## E8002
- Bottleneck hypothesis: rows>=7 failures are geometry/free-param/site-mapping failures after symbolic hits.
- Changed axis: denoising objective and corruption distribution.
- Why positives could appear: direct generative correction can create structures absent from old native candidates.
- Expected metric: rows>=7 new positives on both grouped folds.
- Failure route: if denoiser averages, move to Model C multimodal objective.

## E8009
- Bottleneck hypothesis: the existing MiniCFJoint-v2 path could not be used for opentry_5 terminal metrics because it orders candidates by beam/logprob score.
- Changed axis: inference protocol and joint W/A+geometry generation.
- Why positives could appear: a single-path decoder with exact-cover masks may produce native CIFs in fixed generation order without score-based selection.
- Difference vs failed methods: no selector, no reranker, no beam-score ordering, no source-prior insertion, and no oracle replacement.
- Expected metric: non-oracle match@1/5/20 and rows>=7 positive-any on fold_a.
- Failure route: if fixed-order generation has near-zero W/A-hit or validity, scale training only after fold_b confirms the same failure mode; otherwise change objective/architecture.

## E8010
- Bottleneck hypothesis: any apparent fixed-order gain must be directionally consistent across grouped folds, not isolated to fold_a.
- Changed axis: grouped fold replication of the E8009 fixed-order protocol.
- Why positives could appear: the same no-ranking decoder may generalize to a second prototype-held-out fold.
- Difference vs failed methods: same fixed generation_index/seed order as E8009; no candidate pool sorting.
- Expected metric: fold_b match@1/5/20, rows>=7 positive-any, and W/A-hit match-fail.
- Failure route: if fold_b does not replicate fold_a direction, stop this small smoke branch and either train a stronger full model on dev_model or move to a multimodal objective.

## E8011
- Bottleneck hypothesis: E8009 may be underfit because it used only 128 clean train samples; increasing train coverage may improve native W/A and geometry without any candidate ordering change.
- Changed axis: training data scale and model capacity within the same fixed-order decoder family.
- Why positives could appear: more train prototypes should improve legal W/A coverage and reduce simple geometry failures before changing architecture.
- Difference vs failed methods: no selector, no reranker, no beam-score sorting, no source-prior insertion, and no val512/test access.
- Expected metric: fold_a match@1/5/20 and rows>=7 WA/skeleton hit improve over E8009 while keeping candidate order fixed.
- Failure route: if rows>=7 match remains 0, scale alone is insufficient; move to rows>=7 curriculum or multimodal/free-param objective.

## E8012
- Bottleneck hypothesis: any scale benefit from E8011 must replicate on fold_b before dev_gate.
- Changed axis: grouped fold replication of the scaled fixed-order Model D smoke.
- Why positives could appear: additional train coverage may improve fold_b W/A validity and geometry enough to reduce W/A-hit match-fail.
- Difference vs failed methods: same fixed generation_index/seed protocol as E8011; no candidate ranking or oracle selection.
- Expected metric: fold_b match@1/5/20 and rows>=7 WA/skeleton hit improve over E8010.
- Failure route: if rows>=7 match remains 0 on both scaled folds, stop plain scale-up and change representation/objective.

## E8013
- Bottleneck hypothesis: E8011/E8012 failed rows>=7 partly because the selected 512 train examples contained only 14 rows>=7 structures, while full clean train contains 147.
- Changed axis: rows>=7 data curriculum and oversampling, not inference order.
- Why positives could appear: exposing the decoder to many more complex W/A patterns may improve rows>=7 skeleton/WA coverage under the same fixed-seed generation protocol.
- Difference vs failed methods: not a larger candidate pool, selector, scorer, or reranker; only the training data curriculum changes.
- Expected metric: fold_a rows>=7 WA/skeleton hit and ideally rows>=7 match@20 become nonzero.
- Failure route: if rows>=7 W/A remains zero, curriculum alone is insufficient and the objective/architecture must change.

## E8014
- Bottleneck hypothesis: any rows>=7 curriculum benefit from E8013 must replicate on fold_b before it can be considered directionally consistent.
- Changed axis: grouped fold replication of the rows>=7 curriculum.
- Why positives could appear: the same complex-structure curriculum may improve fold_b rows>=7 W/A/geometry coverage.
- Difference vs failed methods: same no-ranking generation order; no post-generation selection or score sorting.
- Expected metric: fold_b rows>=7 WA/skeleton hit and rows>=7 match@20 improve over E8012.
- Failure route: if both curriculum folds still have rows>=7 match@20 = 0, stop MiniCFJoint cross-entropy curriculum and move to a multimodal/free-param objective or rows>=7-specific architecture.

## E8015
- Bottleneck hypothesis: E8013/E8014 improved rows>=7 W/A/skeleton coverage but left many W/A-hit candidates StructureMatcher-negative because geometry/free-param/collision loss is underweighted.
- Changed axis: training objective weights for free parameters and lattice while keeping the rows>=7 curriculum.
- Why positives could appear: stronger geometry supervision may convert rows>=7 W/A hits into StructureMatcher positives without changing candidate order.
- Difference vs failed methods: no candidate scoring or selection; the model objective changes before generation.
- Expected metric: fold_a rows>=7 match@20 and W/A-hit match-fail improve over E8013.
- Failure route: if rows>=7 positives do not improve, move to a multimodal or denoising geometry model instead of further CE weight scans.

## E8016
- Bottleneck hypothesis: geometry-heavy rows>=7 curriculum must replicate on fold_b before being considered directionally consistent.
- Changed axis: grouped fold replication of E8015.
- Why positives could appear: the fold_b rows>=7 W/A/skeleton hits from E8014 may become matches with stronger geometry/free-param supervision.
- Difference vs failed methods: fixed generation_index/seed order remains unchanged; no score sorting or oracle replacement.
- Expected metric: fold_b rows>=7 match@20 becomes nonzero and W/A-hit match-fail decreases.
- Failure route: if fold_b still has rows>=7 match@20 = 0, stop weighted CE variants and move to a multimodal/free-param generator.
