# OOF W/A Quality Report

Created: 2026-06-19T02:47:58Z

Method: deterministic prototype-frequency W/A predictor. Train rows use 5-fold out-of-fold fits; dev rows use train_core only. This is a condition-gap dataset, not a candidate ranker.

```json
{
  "dev": {
    "full": {
      "fallback_levels": {
        "global": 1,
        "sg": 80,
        "sg_anon": 715,
        "sg_anon_rows": 9438,
        "sg_rows": 1056
      },
      "missing_prediction": 0,
      "samples": 11290,
      "skeleton_exact": 0.17661647475642162,
      "wa_exact": 0.021346324180690875
    },
    "rows_ge_7": {
      "fallback_levels": {
        "sg": 52,
        "sg_anon": 521,
        "sg_anon_rows": 930,
        "sg_rows": 639
      },
      "missing_prediction": 0,
      "samples": 2142,
      "skeleton_exact": 0.18160597572362278,
      "wa_exact": 0.026143790849673203
    }
  },
  "train": {
    "full": {
      "fallback_levels": {
        "global": 16,
        "sg": 328,
        "sg_anon": 1096,
        "sg_anon_rows": 37402,
        "sg_rows": 2495
      },
      "missing_prediction": 0,
      "samples": 41337,
      "skeleton_exact": 0.42690567772213756,
      "wa_exact": 0.017514575319931298
    },
    "rows_ge_7": {
      "fallback_levels": {
        "global": 9,
        "sg": 218,
        "sg_anon": 684,
        "sg_anon_rows": 6465,
        "sg_rows": 1651
      },
      "missing_prediction": 0,
      "samples": 9027,
      "skeleton_exact": 0.44422288689487094,
      "wa_exact": 0.03423064140910601
    }
  }
}
```
