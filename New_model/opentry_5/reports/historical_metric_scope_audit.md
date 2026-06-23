# Historical Metric Scope Audit

Created: 2026-06-19T02:47:58Z

Historical values are not mixed into opentry_5 model selection. They are recorded only to define scope and avoid comparing incompatible E423/E424/E718/aligned55/aligned216/first64/val512 numbers.

Reference target:

- GT-SG CrystaLLM baseline: match@1=26.64%, match@5=36.58%, match@20=44.69%.
- opentry_5 +5pp targets: match@1>=31.64%, match@5>=41.58%, match@20>=49.69%.

Historical E718 full-test aggregate imported in opentry_4:

```json
{
  "crystallm_gt_sg_baseline": {
    "match@1": 0.2664,
    "match@20": 0.4469,
    "match@5": 0.3658
  },
  "e423_val512_baseline": {
    "budget_raw_metrics": {
      "1": {
        "bench_hard_timeout_s": 60.0,
        "bench_max_sites": 512,
        "bench_rmsd_timeout_s": 5.0,
        "bench_workers": 24,
        "match_errors": 0,
        "match_hard_timeouts": 0,
        "match_rate": 0.330078125,
        "match_skipped_large": 0,
        "match_timeouts": 0,
        "n_attempted_candidates": 506,
        "n_ids": 512,
        "parse_rate_candidate": 1.0,
        "rms_dist": 0.10946259560386834,
        "sensible_rate_any": 0.98828125,
        "sensible_rate_candidate": 1.0,
        "valid_rate_any": 0.89453125,
        "valid_rate_candidate": 0.9051383399209486
      },
      "20": {
        "bench_hard_timeout_s": 60.0,
        "bench_max_sites": 512,
        "bench_rmsd_timeout_s": 5.0,
        "bench_workers": 24,
        "match_errors": 0,
        "match_hard_timeouts": 0,
        "match_rate": 0.455078125,
        "match_skipped_large": 0,
        "match_timeouts": 0,
        "n_attempted_candidates": 9717,
        "n_ids": 512,
        "parse_rate_candidate": 1.0,
        "rms_dist": 0.1308637602926111,
        "sensible_rate_any": 0.98828125,
        "sensible_rate_candidate": 1.0,
        "valid_rate_any": 0.89453125,
        "valid_rate_candidate": 0.8786662550169806
      },
      "5": {
        "bench_hard_timeout_s": 60.0,
        "bench_max_sites": 512,
        "bench_rmsd_timeout_s": 5.0,
        "bench_workers": 24,
        "match_errors": 0,
        "match_hard_timeouts": 0,
        "match_rate": 0.408203125,
        "match_skipped_large": 0,
        "match_timeouts": 0,
        "n_attempted_candidates": 2530,
        "n_ids": 512,
        "parse_rate_candidate": 1.0,
        "rms_dist": 0.12014039670083067,
        "sensible_rate_any": 0.98828125,
        "sensible_rate_candidate": 1.0,
        "valid_rate_any": 0.89453125,
        "valid_rate_candidate": 0.9039525691699605
      },
      "50": {
        "bench_hard_timeout_s": 60.0,
        "bench_max_sites": 512,
        "bench_rmsd_timeout_s": 5.0,
        "bench_workers": 24,
        "match_errors": 0,
        "match_hard_timeouts": 0,
        "match_rate": 0.482421875,
        "match_skipped_large": 0,
        "match_timeouts": 0,
        "n_attempted_candidates": 22553,
        "n_ids": 512,
        "parse_rate_candidate": 1.0,
        "rms_dist": 0.13954883644939178,
        "sensible_rate_any": 0.98828125,
        "sensible_rate_candidate": 1.0,
        "valid_rate_any": 0.89453125,
        "valid_rate_candidate": 0.8337250033254999
      }
    },
    "budgets": [
      1,
      5,
      20,
      50
    ],
    "matcher": {
      "angle_tol": 10.0,
      "hard_timeout_seconds": 60.0,
      "ltol": 0.3,
      "max_sites": 512,
      "rmsd_timeout_seconds": 5.0,
      "stol": 0.5,
      "workers": 24
    },
    "rendered_jsonl": "model/New_model/opentry_3/reports/e422_selfscore_e421_chem_vpa_soft_global_val512_top50/rendered_topk_selfscore.jsonl",
    "repr_jsonl": "model/New_model/opentry_3/data/wyckoff_repr_mpts52/val.jsonl",
    "subsets": {
      "atoms_ge_12": {
        "RMSE@1": 0.12200968877156443,
        "RMSE@20": 0.1563864127558895,
        "RMSE@5": 0.14199240828444853,
        "RMSE@50": 0.1709397531878262,
        "W/A@1": 0.44759825327510916,
        "W/A@20": 0.6965065502183406,
        "W/A@5": 0.62882096069869,
        "W/A@50": 0.7096069868995634,
        "candidate_nonempty_rate": 0.9868995633187773,
        "composition_exact_all_candidates@1": 0.9868995633187773,
        "composition_exact_all_candidates@20": 0.9868995633187773,
        "composition_exact_all_candidates@5": 0.9868995633187773,
        "composition_exact_all_candidates@50": 0.9868995633187773,
        "composition_exact_any@50": 0.9868995633187773,
        "match@1": 0.27510917030567683,
        "match@20": 0.3951965065502183,
        "match@5": 0.34934497816593885,
        "match@50": 0.425764192139738,
        "readable_any@50": 0.9868995633187773,
        "samples": 458,
        "skeleton@1": 0.5109170305676856,
        "skeleton@20": 0.740174672489083,
        "skeleton@5": 0.6724890829694323,
        "skeleton@50": 0.7554585152838428,
        "unique_wa_mean@1": 0.9868995633187773,
        "unique_wa_mean@20": 6.746724890829694,
        "unique_wa_mean@5": 2.893013100436681,
        "unique_wa_mean@50": 10.34061135371179
      },
      "complex_flag": {
        "RMSE@1": 0.11933239994889347,
        "RMSE@20": 0.14517373201984102,
        "RMSE@5": 0.13557486176648803,
        "RMSE@50": 0.15880556447994942,
        "W/A@1": 0.4570230607966457,
        "W/A@20": 0.7064989517819706,
        "W/A@5": 0.6352201257861635,
        "W/A@50": 0.7190775681341719,
        "candidate_nonempty_rate": 0.9874213836477987,
        "composition_exact_all_candidates@1": 0.9874213836477987,
        "composition_exact_all_candidates@20": 0.9874213836477987,
        "composition_exact_all_candidates@5": 0.9874213836477987,
        "composition_exact_all_candidates@50": 0.9874213836477987,
        "composition_exact_any@50": 0.9874213836477987,
        "match@1": 0.2914046121593291,
        "match@20": 0.4192872117400419,
        "match@5": 0.3752620545073375,
        "match@50": 0.44863731656184486,
        "readable_any@50": 0.9874213836477987,
        "samples": 477,
        "skeleton@1": 0.5241090146750524,
        "skeleton@20": 0.7484276729559748,
        "skeleton@5": 0.6834381551362684,
        "skeleton@50": 0.7631027253668763,
        "unique_wa_mean@1": 0.9874213836477987,
        "unique_wa_mean@20": 7.0083857442348005,
        "unique_wa_mean@5": 2.911949685534591,
        "unique_wa_mean@50": 10.985324947589099
      },
      "full": {
        "RMSE@1": 0.10946259560386833,
        "RMSE@20": 0.1308637602926112,
        "RMSE@5": 0.12014039670083074,
        "RMSE@50": 0.13954883644939187,
        "W/A@1": 0.482421875,

```

opentry_4 train-dev/val512 aggregate summary:

```json
{
  "anchor": {
    "anchor_safe_selected_rows_ge_7": {
      "RMSE@1": 0.09913643174933533,
      "RMSE@20": 0.1340250749335289,
      "RMSE@5": 0.11449343424811396,
      "RMSE@50": 0.1340250749335289,
      "W/A@1": 0.45701357466063347,
      "W/A@20": 0.6787330316742082,
      "W/A@5": 0.6470588235294118,
      "W/A@50": 0.6787330316742082,
      "match@1": 0.13574660633484162,
      "match@20": 0.19004524886877827,
      "match@5": 0.17194570135746606,
      "match@50": 0.19004524886877827,
      "samples": 221,
      "skeleton@1": 0.5113122171945701,
      "skeleton@20": 0.755656108597285,
      "skeleton@5": 0.7058823529411765,
      "skeleton@50": 0.755656108597285
    },
    "anchor_safe_selected_val512": {
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
    },
    "baseline_e424_order_rows_ge_7": {
      "RMSE@1": 0.09913643174933533,
      "RMSE@20": 0.1340250749335289,
      "RMSE@5": 0.11449343424811396,
      "RMSE@50": 0.1340250749335289,
      "W/A@1": 0.45701357466063347,
      "W/A@20": 0.6787330316742082,
      "W/A@5": 0.6470588235294118,
      "W/A@50": 0.6787330316742082,
      "match@1": 0.13574660633484162,
      "match@20": 0.19004524886877827,
      "match@5": 0.17194570135746606,
      "match@50": 0.19004524886877827,
      "samples": 221,
      "skeleton@1": 0.5113122171945701,
      "skeleton@20": 0.755656108597285,
      "skeleton@5": 0.7058823529411765,
      "skeleton@50": 0.755656108597285
    },
    "baseline_e424_order_val512": {
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
    },
    "merged_full_val512_ceiling": {
      "positive_any_all_candidates": 0.4644268774703557,
      "positive_samples": 235,
      "samples": 506
    },
    "merged_rows_ge_7_ceiling": {
      "positive_any_all_candidates": 0.19909502262443438,
      "positive_samples": 44,
      "samples": 221
    },
    "proposal_direct_order_aligned216": {
      "RMSE@1": 0.20623964427311994,
      "RMSE@20": 0.2346528518397651,
      "RMSE@5": 0.2220897278103643,
      "RMSE@50": 0.2346528518397651,
      "W/A@1": 0.5046296296296297,
      "W/A@20": 0.6898148148148148,
      "W/A@5": 0.6851851851851852,
      "W/A@50": 0.6898148148148148,
      "match@1": 0.09259259259259259,
      "match@20": 0.13425925925925927,
      "match@5": 0.12037037037037036,
      "match@50": 0.13425925925925927,
      "samples": 216,
      "skeleton@1": 0.5185185185185185,
      "skeleton@20": 0.7638888888888888,
      "skeleton@5": 0.7361111111111112,
      "skeleton@50": 0.7638888888888888
    }
  },
  "joint_generator": {
    "baseline_same_samples": {
      "RMSE@1": 0.08173795273282784,
      "RMSE@20": 0.04879829767169925,
      "RMSE@5": 0.02984752389098976,
      "RMSE@50": 0.04879829767169925,
      "W/A@1": 0.5384615384615384,
      "W/A@20": 0.6153846153846154,
      "W/A@5": 0.5769230769230769,
      "W/A@50": 0.6153846153846154,
      "match@1": 0.34615384615384615,
      "match@20": 0.4230769230769231,
      "match@5": 0.38461538461538464,
      "match@50": 0.4230769230769231,
      "rows": 468,
      "samples": 26,
      "skeleton@1": 0.6538461538461539,
      "skeleton@20": 0.7307692307692307,
      "skeleton@5": 0.6538461538461539,
      "skeleton@50": 0.7307692307692307
    },
    "merged_positive_any": {
      "positive_any_all_candidates": 0.4230769230769231,
      "positive_samples": 11,
      "samples": 26
    },
    "new_positive_sample_ids": [],
    "new_positive_samples_beyond_baseline": 0,
    "proposal_direct": {
      "RMSE@1": 0.15324333270966373,
      "RMSE@20": 0.17328691056997114,
      "RMSE@5": 0.19327227020979015,
      "RMSE@50": 0.17328691056997114,
      "W/A@1": 0.34615384615384615,
      "W/A@20": 0.6153846153846154,
      "W/A@5": 0.5769230769230769,
      "W/A@50": 0.6153846153846154,
      "match@1": 0.15384615384615385,
      "match@20": 0.34615384615384615,
      "match@5": 0.3076923076923077,
      "match@50": 0.34615384615384615,
      "rows": 208,
      "samples": 26,
      "skeleton@1": 0.38461538461538464,
      "skeleton@20": 0.7307692307692307,
      "skeleton@5": 0.7307692307692307,
      "skeleton@50": 0.7307692307692307
    }
  },
  "val512": {
    "anchor_safe_selected_rows_ge_7": {
      "RMSE@1": 0.09913643174933533,
      "RMSE@20": 0.1340250749335289,
      "RMSE@5": 0.11449343424811396,
      "RMSE@50": 0.1340250749335289,
      "W/A@1": 0.45701357466063347,
      "W/A@20": 0.6787330316742082,
      "W/A@5": 0.6470588235294118,
      "W/A@50": 0.6787330316742082,
      "match@1": 0.13574660633484162,
      "match@20": 0.19004524886877827,
      "match@5": 0.17194570135746606,
      "match@50": 0.19004524886877827,
      "samples": 221,
      "skeleton@1": 0.5113122171945701,
      "skeleton@20": 0.755656108597285,
      "skeleton@5": 0.7058823529411765,
      "skeleton@50": 0.755656108597285
    },
    "anchor_s
```

Audit decision: opentry_5 will report historical E423/E718/current no-ranking baseline/new-model metrics as separate rows. No full-test rerun is authorized here.
