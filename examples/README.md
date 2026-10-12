# Examples: one real night, every file

Night 12 (Sunday 2026-10-11 into Monday) tested the pooling of the two best CNNs: 2 sweeps, 200 jobs, then the morning analysis. 06_example_night.md tells it as a story; these are the files themselves.

| Path | What it is |
|---|---|
| night12_prep/README.txt | what was prepared before the night and how each part was checked |
| night12_prep/make_night12.py, make_night12.log | writes the two sweep files from the finalists' records and checks them with the repository's sweep expansion |
| night12_prep/RULE_night12_tests.txt | which runs, which test files and which comparisons, fixed before any result existed |
| night12_prep/lines/ | the exact lines Harrison pasted, in order. Lines 1 and 5 use his PC's push.ps1 and pull.ps1; with your own clone from GitHub you update with git instead (04_setup_from_raw_data.md) |
| night12_analysis/README.txt | the morning, step by step with times: validation files, analyze, a repeat check, the selection records, then the test files and the test comparisons |
| night12_analysis/*.py | the morning's scripts (selection records, repeat check, test families) |
| night12_analysis/*.txt, *.json | their outputs: analyze reports, the printed records, the test reports, the families N12A and N12B |
| first_run_expected_output.txt | what analyze and test-report print for the first run's four configurations (05_first_run.md); 05's expected numbers come from it |
