# Corruption Dataset Report

Created: 2026-06-19T02:47:58Z

Corruption proportions follow the opentry_4 diagnostic priority: free-param/site-mapping about 50%, collision/short-contact about 34%, lattice/VPA about 9%, inter-row distance about 7%.

```json
{
  "dev": {
    "by_corruption_type": {
      "collision_or_short_contact_noise": 1316,
      "free_param_or_site_mapping_noise": 2030,
      "inter_row_distance_noise": 227,
      "lattice_vpa_noise": 368
    },
    "by_source": {
      "synthetic_train_dev_clean_corruption": 3941
    },
    "examples": 3941,
    "rows_ge_7": 2142
  },
  "train": {
    "by_corruption_type": {
      "collision_or_short_contact_noise": 5290,
      "free_param_or_site_mapping_noise": 7650,
      "inter_row_distance_noise": 943,
      "lattice_vpa_noise": 1471,
      "real_train_hard_negative_wa_or_skeleton_hit_match_fail": 2613
    },
    "by_source": {
      "opentry3_e318_train_hard_negative": 2382,
      "opentry4_pairfield_train_hard_negative": 231,
      "synthetic_train_dev_clean_corruption": 15354
    },
    "examples": 17967,
    "rows_ge_7": 9869
  }
}
```

Sources:

- train/dev synthetic corruptions from canonical train/dev only;
- train hard negatives from opentry_3 E318 and opentry_4 E7010 where W/A or skeleton is hit but StructureMatcher fails.

No val512 positives or test data are used.
