# No-Ranking Baseline Report

Created: 2026-06-19T02:47:58Z

Current available no-new-run baseline evidence comes from the frozen opentry_4 aggregate files. It is kept separate from new opentry_5 grouped-dev metrics because those historical runs used their own sample scopes.

Baseline target for opentry_5:

- Before model training, the canonical grouped-dev evaluator must consume candidates in fixed native `generation_index` order.
- If a CrystaLLM native K=20 run is launched, generation_index=0 is deterministic decode, 1-4 fixed medium-random seeds, and 5-19 fixed diversity seeds.
- Invalid candidates are counted in place.

Known historical val512 fixed-order aggregate from opentry_4 E424 scope:

```json
{
  "RMSE@1": 0.10946259560386833,
  "RMSE@20": 0.16897342100460178,
  "RMSE@5": 0.14795029036576465,
  "RMSE@50": 0.16897342100460178,
  "W/A@1": 0.4881422924901186,
  "W/A@20": 0.7272727272727273,
  "W/A@5": 0.6561264822134387,
  "W/A@50": 0.7272727272727273,
  "match@1": 0.3339920948616601,
  "match@20": 0.46047430830039526,
  "match@5": 0.41304347826086957,
  "match@50": 0.46047430830039526,
  "samples": 506,
  "skeleton@1": 0.5553359683794467,
  "skeleton@20": 0.7707509881422925,
  "skeleton@5": 0.7055335968379447,
  "skeleton@50": 0.7707509881422925
}
```

Status: canonical split/evaluator is now frozen; new model training remains gated on this protocol. This file does not claim a new opentry_5 no-ranking baseline run.
