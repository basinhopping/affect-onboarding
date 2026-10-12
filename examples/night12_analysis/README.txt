Night 12's morning analysis (container): the pooling of the two finalists, as RULE_night12_tests.txt
(analyses/night12_prep_20261012, fixed 2026-10-12 06:27 UTC, amended 06:44 UTC, SHA-256 57ec0996...) says.

Written before any file of night 12 left the node; the morning's results are at the end.

Scripts
  select_night12.py        the 8 rows' selection records (each finalist's recorded settings, and the same with
                           only model.pooling changed); --print first, then for real; a row without all 25 runs
                           (folds test_S1 to test_S5, seeds 6 to 10) gets no record and is listed in
                           unfinished_rows.json
  night12_test_analysis.py families N12A (the tuned 3-block CNN: mean, max and attention pooling, each minus its
                           recorded SD pooling) and N12B (cnn_final_3: SD, max and attention pooling, each minus its
                           recorded mean pooling), Holm over 3 within each; pairing.py of analyses/night9_20261009,
                           generator seed 20261012, 2,000 draws; each row's test UA with its 95% t interval over
                           seeds; writes night12_test.json
  sim_morning/run_sim.py   the dry run: a temporary repository with the code, the two sweep files, the finalists'
                           records and fake run folders (each job's own spec.json with the files of the confirmation's
                           run of the same fold and seed, so every pooling row carries its base's predictions), then
                           every step in the rule's order; with equal predictions every difference must be 0. A
                           second case removes one variant's runs to check the unfinished-row path.

The order in the morning
  1. line 5 (lines/5_pack_night12.txt of the prep folder) on the node, then pull.ps1 on the PC
  2. validation files only to the container (to_cloud.sh val, from_pc.sh, check_staged.py of
     analyses/night6_prep_20260929), checked against the pull's list; no test file
  3. affect.cli analyze of cnn_3block_pooling_1 and cnn_final_3_pooling_1 (confidence 0.95, 2,000 draws, seed
     20261012), then compare-runs of each
  4. select_night12.py --print, then select_night12.py
  5. the test files of the recorded rows' runs only, checked against the pull's list
  6. affect.cli test-report of each record, then night12_test_analysis.py

Changes made before any file of night 12 reached the container (2026-10-12, UTC)
  07:37  night12_test_analysis.py: pairing.N_BOOT 20,000 -> 2,000 and its docstring, as the rule's 06:44 amendment
         says (it had been left at 20,000).
  07:57  select_night12.py: it asserted that each sweep's analyze report holds exactly 4 configurations, but a
         configuration with no complete run is absent from the report, so a variant whose runs all failed would have
         stopped the script instead of being listed as unfinished, as the rule says. Now 1 to 4 configurations, each
         one of the sweep's rows (a configuration in the report that is no row stops the script). Found by the dry
         run's second case (sim_morning/run_sim_output_2000draws_before_fix.txt).

Dry runs
  run_sim_output.txt                      written by 06:32 UTC, 20,000 draws: stopped by the command's time limit
                                          during case 1's test analysis (the reason for the amendment)
  run_sim_output_2000draws_before_fix.txt 07:38 to 07:40 UTC: case 1 passed; case 2 failed in select_night12.py
                                          (above)
  run_sim_output_after_fix.txt            07:58 to 08:03 UTC: both cases passed (case 2: pool_3block_max unfinished,
                                          N12A Holm over 2, every difference 0)

The morning of 2026-10-12 (UTC)
  17:55  line 5 on the node: 200 of 200 jobs done, 0 failed, 0 pending; 1,861 entries packed
         (affect_results_runs_cnn_3block_pooling_1_and_8_more_20261012T175534Z); pull.ps1 on the PC: 1,648 files
         unpacked and checked against the node's hashes.
  18:00  validation files only (to_cloud.sh val, from_pc.sh, check_staged.py): 1,208 files, each matched to the pull's
         list, 0 test files in the container.
  18:01  analyze of both sweeps (analyze_*.txt; reports cnn_3block_pooling_1_20261012T180132_038893Z and
         cnn_final_3_pooling_1_20261012T180140_720505Z). affect.cli compare-runs stopped on the missing test files,
         which it hashes (compare_runs_*.txt), so repeat_check_night12.py made the same comparison: every one of the
         50 runs of the two recorded configurations equals its earlier copy (cnn_3block_confirm_1, cnn_confirm_3, and
         cnn_regression_3 for cnn_final_3's seed 6) in its validation files and, by SHA-256, its test files:
         REPEAT CHECK PASSED (repeat_check_night12.json).
  18:03  select_night12.py --print, then for real: the 8 records, read back; 0 unfinished rows.
  18:03  only then the 400 test files of the 200 runs (to_cloud.sh test, from_pc.sh, check_staged.py: problems 0);
         test-report of the 8 records (test_reports.txt); night12_test_analysis.py (night12_test.json, 2 minutes).

Validation UA, seeds 6 to 10 (analyze):
  tuned 3-block CNN: SD (recorded) 0.6340; max 0.6198 (-0.0142, worse); mean 0.6171 (-0.0168, worse); attention 0.5929
  (-0.0411, worse). cnn_final_3: mean (recorded) 0.6430; SD 0.6420 (-0.0009); max 0.6336 (-0.0094); attention 0.6123
  (-0.0306, worse).

Test UA, session out, seeds 6 to 10, with 95% t intervals over seeds:
  tuned 3-block CNN: SD 0.6010 [0.5920, 0.6100], max 0.5937 [0.5854, 0.6020], mean 0.5756 [0.5603, 0.5910],
                     attention 0.5637 [0.5511, 0.5763]
  cnn_final_3:       SD 0.6083 [0.6013, 0.6152], mean 0.6073 [0.6026, 0.6120], max 0.5999 [0.5936, 0.6062],
                     attention 0.5832 [0.5690, 0.5975]
  N12A (each minus SD pooling, Holm over 3): mean -0.0253 [-0.0372, -0.0142], Holm p 0.0015, worse; max -0.0073
       [-0.0164, +0.0019], Holm p 0.129, no detectable difference; attention -0.0373 [-0.0487, -0.0260], Holm p 0.0015,
       worse.
  N12B (each minus mean pooling, Holm over 3): SD +0.0009 [-0.0061, +0.0082], Holm p 0.777; max -0.0074
       [-0.0172, +0.0026], Holm p 0.288, both no detectable difference; attention -0.0241 [-0.0347, -0.0135], Holm p
       0.0015, worse.
  So SD pooling helps the tuned 3-block CNN and makes no detectable difference in cnn_final_3, and attention pooling is
  worse in both. Holm p 0.0015 is the smallest a family of 3 can show with 2,000 draws.
  The meeting doc (paragraph "Tested directly on both finalists.", revision 69) and the prep sheet carry these numbers.
