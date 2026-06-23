# CrystaLLM Validation Anchor Partial Report

- Created at: 2026-06-22T17:06:28+00:00
- Scope: completed opentry_9 validation GT-SG anchor shards only.
- Candidate budget used for this report: K20, ranks 1-20 from the historical CrystaLLM-a GT-SG sampling policy.
- This is not a full validation gate and must not be used to freeze a strategy.

| dataset | samples | candidates | match@1 | match@5 | match@20 | rows>=7 samples | rows>=7 match@1 | rows>=7 match@5 | rows>=7 match@20 | top1 fail but top20 hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mp_20 | 23 | 460 | 0.0000 | 0.0000 | 0.0000 | 23 | 0.0000 | 0.0000 | 0.0000 | 0 |
| mpts_52 | 8 | 160 | 0.0000 | 0.0000 | 0.0000 | 8 | 0.0000 | 0.0000 | 0.0000 | 0 |
