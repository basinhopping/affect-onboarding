# First run: the baseline CNN on the four class sets

A short run to learn every step on real data: the baseline 4-block CNN under the four class sets the advisor asked about, on the five session-out folds, seed 1 only. Four sweeps of 5 jobs, 20 jobs in all, about a minute each (the baseline stops after about 19 to 26 epochs), so about 10 minutes on one GPU. Run it in the evening, like every GPU job.

Each sweep repeats the seed-1 runs of a configuration Harrison already ran (same settings, folds and seed), so the results can be checked against his to the last digit when the package versions match (04_setup_from_raw_data.md, last section).

| Sweep | Class set | Configuration | Harrison's record |
|---|---|---|---|
| intro_base_four | neutral, happy+excited, angry, sad | cf0495a296e9 | cnn_base_session_out |
| intro_base_fru5 | the four plus frustrated | 5f51b5b0e9ff | base_classes_fru5 |
| intro_base_fru4 | frustrated in place of neutral | d3fe88422e3a | base_classes_fru4 |
| intro_base_no_neu3 | happy+excited, angry, sad | babf12b6113e | base_classes_no_neu3 |

The four sweep files are in `sweeps/` of the repository, and a copy is in this folder's `sweeps/`. If `ls ~/affect/sweeps/intro_base_*` on the node lists none, copy them from this folder on your computer: `scp sweeps/intro_base_*.json jcarnegie@seas-gpu-node-01:~/affect/sweeps/`.

## The lines, in order (on the node)

1. Which GPUs are free: pick one at 0 % with about 70 MiB in use (GPU 0 below; change both 0s otherwise).

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

2. Smoke-test and queue the four sweeps (a few minutes). It should print `SMOKE OK` for each, then `ALL DONE <time>, pending 20`:

```bash
cd ~/affect && SMOKE_GPU=0 bash scripts/smoke_enqueue.sh intro_base_four intro_base_fru5 intro_base_fru4 intro_base_no_neu3
```

3. Start three workers on that GPU. They exit by themselves when the queue is empty:

```bash
rm -f ~/affect/queue/STOP; tmux new -d -s gpu0 "bash ~/affect/scripts/worker.sh 0 3"; sleep 60; tmux ls; echo "running $(ls ~/affect/queue/running | wc -l), pending $(ls ~/affect/queue/pending | wc -l), done $(ls ~/affect/queue/done | wc -l)"
```

4. Follow progress (any time; when `tmux ls` no longer lists gpu0, the workers are done):

```bash
tmux ls; echo "running $(ls ~/affect/queue/running | wc -l), pending $(ls ~/affect/queue/pending | wc -l), done $(ls ~/affect/queue/done | wc -l), failed $(ls ~/affect/queue/failed | wc -l)"; bash ~/affect/scripts/affect.sh status --sweep intro_base_four --top 5
```

5. When all 20 are done: compare on validation (with one configuration a sweep, this just summarizes it). Expect validation UA (seed 1) 0.5792, 0.4728, 0.5486 and 0.7235:

```bash
cd ~/affect && for s in intro_base_four intro_base_fru5 intro_base_fru4 intro_base_no_neu3; do bash scripts/affect.sh analyze --sweep $s --confidence 0.95 --n-boot 2000 --seed 20261012 --top 1; done
```

6. Record the choice before reading any test file (with one configuration a sweep, nothing is really chosen, but the order is the habit):

```bash
cd ~/affect && bash scripts/affect.sh select --sweep intro_base_four --config cf0495a296e9 --name intro_four --reason "First run: the baseline CNN under four classes, seed 1, to learn the steps" && bash scripts/affect.sh select --sweep intro_base_fru5 --config 5f51b5b0e9ff --name intro_fru5 --reason "First run: the baseline CNN under five classes, seed 1, to learn the steps" && bash scripts/affect.sh select --sweep intro_base_fru4 --config d3fe88422e3a --name intro_fru4 --reason "First run: the baseline CNN under four classes without neutral, seed 1, to learn the steps" && bash scripts/affect.sh select --sweep intro_base_no_neu3 --config babf12b6113e --name intro_no_neu3 --reason "First run: the baseline CNN under three classes, seed 1, to learn the steps"
```

7. Read the test results. Expect test UA 0.5520, 0.4345, 0.5167 and 0.6873 (Harrison's seed-1 values):

```bash
cd ~/affect && for n in intro_four intro_fru5 intro_fru4 intro_no_neu3; do bash scripts/affect.sh test-report --selection $n --confidence 0.95; done
```

Each report prints UA, WA, macro-F1, each class's recall, UA by speaker gender and by improvised or scripted speech, and the UA of every fold. For four classes the folds read 0.5941, 0.5592, 0.4926, 0.5116 and 0.6026 in Harrison's run.

## What to look at afterwards

- One run folder, for example `~/affect/artifacts/runs/intro_base_four/cf0495a296e9_test_S1_seed1/`: `log.txt` shows one line per epoch with the loss and validation UA, and which epoch was kept; `efficiency.json` the epochs run and seconds.
- Why the numbers fall with five classes (0.4345) and rise with three (0.6873): chance is 0.20 and 0.33 there, against 0.25 with four; frustrated sits between the other classes.
- How far one seed is from five: Harrison's five-seed means for the same configurations are 0.559, 0.439, 0.521 and 0.683.

## How this differs from the real experiments

This run is 20 jobs: one configuration a class set, one seed. A result the project reports uses five seeds (25 runs a configuration), and the configurations themselves came out of screens of hundreds of configurations and confirmations on new seeds: about 37,000 runs and 1.8 million training epochs so far (07_scale_of_the_work.md). The same commands run all of it; only the sweep files grow.
