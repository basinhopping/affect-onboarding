# The scale of the work so far

Counted on 2026-10-12 from every run folder pulled back from the node (each run's `efficiency.json`), from night 1 (2026-09-25) to night 11 (2026-10-11). Night 12 then added 200 more (200 done, 0 failed).

| Kind of work | Sweeps | Runs | Training epochs | Job-hours |
|---|---|---|---|---|
| Tuning the 4-block CNN (screens, grids, confirmations) | 16 | 15,775 | 752,710 | 746 |
| Other splits and subsets (the 27-split study) | 84 | 8,370 | 398,924 | 217 |
| CNN + Transformer | 3 | 4,565 | 216,728 | 255 |
| CNN + LIF (hybrid and fully spiking) | 18 | 4,325 | 226,584 | 800 |
| The 3-block CNN and its sweep | 16 | 2,810 | 140,891 | 256 |
| Class sets | 28 | 1,000 | 49,048 | 52 |
| Classical baselines (SVM, TPOT), on CPUs | 3 | 295 | 0 | 24 |
| Training-accuracy and session-out rows | 5 | 125 | 6,760 | 7 |
| Reproducibility checks (re-runs that must match earlier runs) | 4 | 70 | 2,836 | 3 |
| **Total** | **177** | **37,335** | **1,794,481** | **2,361** |

- A run is one configuration trained on one fold with one seed. The 37,040 GPU runs averaged 48.4 epochs each; the longest configurations allow 100.
- Job-hours add up each job's own training and test time. Jobs shared each GPU three at a time, on four GPUs, mostly at night.
- Most runs are search: screens try hundreds of configurations on validation (the 3-block sweep alone was 1,960 runs), the best are confirmed on five new seeds, and only the chosen ones are read on test.

## Next to the first run

05_first_run.md is 20 runs: one configuration under each of four class sets, one seed, about a minute and 19 to 26 epochs each, so about 20 job-minutes. One result the project reports (one configuration, five seeds, five folds) is 25 runs; the CNN that reached 0.607 was chosen from about 16,000 runs of tuning. The commands are the same at both sizes; what changes is how many configurations and seeds a sweep holds, and how carefully the comparisons are planned.
