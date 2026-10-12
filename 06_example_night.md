# Example: one night of experiments, start to finish

This is how a real night runs, with the lines and the output Harrison pasted back. Every file of it is in examples/: night12_prep/ (the sweep maker, the RULE, the lines) and night12_analysis/ (the morning's scripts and outputs). Night 12 (Sunday 2026-10-11 into Monday) tested the pooling of the two best CNNs: 2 sweeps, 200 jobs. Night 11's analysis, below, shows the morning half of the cycle with its results.

## Evening: prepare (Claude, before anything runs)

1. Write the sweep files with a script that checks them: here `make_night12.py` read the two finalists' selection records, changed only `model.pooling`, and checked with the repository's sweep expansion that each sweep holds 100 jobs, that each base hashes to its record, and that each variant differs in pooling only. The sweep tests (`tests/test_sweep.py`) passed with the new files.
2. Write the RULE file before any result exists: which runs are made, which test files will be read, which comparisons will be made (families of comparisons with Holm's correction), and what happens to unfinished rows. It is saved with the time it was fixed.
3. Write each line the person will paste, check its bash syntax, and run the start line on a simulated node (fake tmux and nvidia-smi) to see it start the right GPUs and refuse when it should.
4. Compute the code tree's digest (a SHA-256 over every code file), so the node can confirm it received exactly this code.

## Evening: run (the person pastes, Claude checks each output)

Line 1, on the PC (PowerShell): send the code to the node.

```
powershell -ExecutionPolicy Bypass -File "C:\Affective Computing\affect\scripts\push.ps1"
```

Line 2, on the node: unpack, confirm the digest, run every test and check the data and GPUs. Output (the log's path left out, the home folder shown as ~):

```
affect_code_20261012T063223Z.tar.gz: OK
code in ~/affect replaced from affect_code_20261012T063223Z.tar.gz
TREE OK 5a49cdcee1eca39c9d1d64b3cc079f45638506bfc1a3507b2a38d72870a31b5b
Ran 202 tests in 127.297s
OK
check: all three passed.
check.sh exit 0
```

Line 3, on the node: refuse if workers or jobs exist; print the GPUs' use; smoke-test and queue both sweeps on GPU 0; then start 3 workers on every GPU no other user is on, with a schedule step that stops everything at 08:00. Output:

```
0, 0 %, 70 MiB
1, 0 %, 70 MiB
2, 0 %, 70 MiB
3, 0 %, 70 MiB
night12 started: smoke tests on GPU 0 and queueing first (about 10 to 20 minutes), then 3 workers on every GPU no one else is using, until the queue is empty or 08:00 Pacific
```

Line 4, about 10 minutes later: status. Output (shortened):

```
ALL DONE 2026-10-12T06:36:25Z, pending 200
gpu0 ... gpu3: 1 windows each; night12: 1 windows
night: GPU 0 started (3 workers, tmux session gpu0) ... GPU 3 started
running 12, pending 185
cnn_3block_pooling_1: done 3, running 12, pending 85, failed 0
cnn_final_3_pooling_1: done 0, running 0, pending 100, failed 0
0, 100 %, 11463 MiB ... 3, 100 %, 11543 MiB
```

Twelve jobs at a time (3 per GPU), all four GPUs at 100 %, nothing failed: a healthy night.

## Morning: bring the results back and analyze (night 11 as the example)

1. Line 5, on the node: refuse while workers run; count each sweep's jobs; copy the night's logs; pack the run folders and sweep records into one archive with a SHA-256 list of every file. Then, on the PC, `pull.ps1 <archive name>` downloads it, checks every file against the node's list and unpacks it into `artifacts/`. Night 11's pull: 4,515 files checked.
2. Validation files first: they are staged into Claude's workspace and checked by hash; no test file comes along yet. `analyze` runs on each sweep.
3. The selection records are written (here one per row, as the RULE fixed them), and read back.
4. Only then the test files of exactly those rows are staged and checked, `test-report` runs for each record, and the families of comparisons run as the RULE says (paired bootstrap, Holm).
5. Everything goes into `artifacts/analyses/night11_20261011/` with a README: what was run, what was read when, and the results. Night 12's own morning is in examples/night12_analysis/README.txt, step by step with times.

Night 11's results (pooled folds, seeds 1 to 5, test UA): cnn_final_3 0.678; CNN + LIF hybrid 0.662; the tuned 3-block CNN 0.654; fully spiking 0.615; the baseline 0.602; the 3-block CNN as specified 0.585. Family N9F, for example: the hybrid minus the baseline +0.060 (Holm-adjusted p 0.002, better), the hybrid minus cnn_final_3 -0.016 (Holm-adjusted p 0.006, worse).

## The habits this shows

- Nothing runs that was not checked first, and nothing is read from test that was not decided first.
- Every line says what it should print; the person pastes the output and Claude compares.
- Every file that crosses between machines is checked by SHA-256.
- A night ends by itself (empty queue or the stop step), so a forgotten run cannot hold the GPUs into the day.
