# Next Action

1. Current frozen candidate is E718 val512 GT-free apply path.
2. E718 `match@1/5/20/50=33.01/41.99/45.90/48.24%`; RMSE `0.1095/0.1281/0.1318`.
3. E718 clears MPTS-52 +5pp targets for match@1 and match@5; match@20 still misses.
4. Rows>=7 K5/K20/K50: `18.22/19.11/20.44%`; not worse than E423.
5. No-leakage audit: `reports/e718_freeze_no_leakage_audit/frozen_config_audit.md`.
6. Full test has not run; one frozen full-test attempt is now allowed.
7. Missing artifact: E420/E421/E422-style full MPTS-52 test candidate pool under opentry_3.
8. Generate full test W/A predictions with the frozen E416-E420 recipe; do not use test W/A labels.
9. Render with frozen E421 config, selfscore top50, score with E425 model via `opentry_score_rendered_candidates_apply.py`.
10. Apply frozen selector: threshold `0.0024707304479371964`, anchor `4`, max_per_wa `2`, top_k `50`.
11. Evaluate once on full MPTS-52 test budgets `1,5,20,50`.
12. Do not alter config using full-test feedback.
