# The affect repository: a guide

The code is at https://github.com/basinhopping/affect (public). Line numbers below are those of commit d193f4e, the version pushed on 2026-10-11. The repository's own README.md describes the layout; its push and pull steps are for Harrison's PC and do not apply to a clone on the node.

## One experiment, start to finish

1. **A sweep file** (`sweeps/<sweep_id>.json`) holds every setting of a base configuration, the folds, and stages of configurations to try: the base, named variants, a grid, one change at a time, or random combinations.
2. **smoke** (`affect.sh smoke`) tries every configuration for one training step and runs one short real job. It must end with `SMOKE OK`.
3. **enqueue** expands the sweep into one job file per configuration, fold and seed in `~/affect/queue/pending/`, and records the sweep in `artifacts/sweeps/<sweep_id>/`.
4. **Workers** (`scripts/worker.sh <gpu> <processes>`, each in a tmux session) take the highest-priority pending job, run it, and move its file to `queue/done/` or `queue/failed/`, until the queue is empty or `queue/STOP` exists.
5. **One job** (`src/affect/train/loop.py`, `train_job`) loads the audio, computes log-mel features, trains, keeps the epoch with the best validation UA, stops after `early_stopping_patience` epochs without a higher one, and predicts validation and test once with the kept weights. It writes `artifacts/runs/<sweep_id>/<run_id>/`.
6. **analyze** compares every configuration with the sweep's base on validation files only (paired bootstrap, Holm), and writes `artifacts/analysis/<sweep_id>_<utc>/`.
7. **select** records the chosen configuration and the reason in `artifacts/selections/<name>/`. It refuses before analyze has run, and never overwrites a record.
8. **test-report** reads the test files of that configuration's runs only: UA, WA, macro-F1 and each class's recall, per seed and fold, in `artifacts/test_reports/<name>_<utc>/`.

## A sweep file, read line by line

`sweeps/intro_base_four.json` (the first run in 05_first_run.md) has five parts:

- `sweep_id`: the name; folders and job files carry it.
- `description`: why the sweep exists, in words.
- `folds`: `test_S1` to `test_S5`, the five session-out folds (fold k tests session k and validates on session k + 1).
- `base`: every setting, in five sections; none has a default, so a file says everything:
  - `data`: the dataset, the label set (`four_merged`, `five_with_frustration`, `four_with_frustration_no_neutral`, `three_no_neutral`, and others), the subset (`all`, `impro`, `script`) and the split (`leave_one_session_out`, `utterance_5fold` for pooled, and 19 more).
  - `features`: the log-mel spectrogram: 25 ms windows every 10 ms, 512-point FFT, 64 mel bands, normalized per utterance.
  - `augment`: SpecAugment masks (all 0 here: off).
  - `model`: the CNN: blocks of 32, 64, 128 and 256 channels, each one 3 by 3 convolution with batch norm and ReLU, then 2 by 2 max pooling; mean pooling over time; dropout 0.3; then the output layer.
  - `train`: up to 60 epochs, batches of 32, AdamW with learning rate 0.001 after 2 warm-up epochs, cosine decay, weight decay 0.0001, patience 10, random 5 s training crops, bf16 precision.
- `stages`: here one, the base, on seed 1.

A configuration's hash (`config_hash`, 64 hex characters; the first 12 name it) is the SHA-256 of its settings. The same settings always give the same hash, so a result can be traced to its exact settings, and two runs of the same configuration, fold and seed should give identical results.

## A run folder

`artifacts/runs/<sweep_id>/<run_id>/`, where `run_id` is `<hash12>_<fold>_seed<seed>`:

- `spec.json`: the job (settings, fold, seed, stage, label).
- `metrics_val.json`: the kept epoch, its validation UA, the per-epoch curves, the full validation report.
- `predictions_val.csv`: every validation utterance with its label, prediction and class probabilities.
- `metrics_test.json`, `predictions_test.csv`: the same for test. Only `test-report` reads them.
- `efficiency.json`: parameters, training seconds, seconds per epoch, epochs run, peak GPU memory.
- `provenance.json`: the code's hash, the data manifest's hash, the folds' hash, package versions, the machine.
- `log.txt`: the job's log, one line per epoch.

## The code

| Folder or file | What it holds |
|---|---|
| `src/affect/data/iemocap.py` | reading IEMOCAP's label files, the label sets, soft labels, the folds |
| `src/affect/data/iemocap_splits.py`, `splits.py` | the 21 split schemes; the pooled deal by speaker and class |
| `src/affect/data/ingest.py`, `assemble.py` | copying the raw folders with manifests; building `data/iemocap/assembled` |
| `src/affect/features/logmel.py` | the log-mel spectrogram |
| `src/affect/augment/` | SpecAugment, speed perturbation, mixup |
| `src/affect/models/cnn.py` | the CNN (used for the baseline, cnn_final_3 and the 3-block CNN; the block count is the length of `channels`) and its pooling (mean, SD, max, attention, lines 98 to 121) |
| `src/affect/models/lif.py`, `cnn_lif.py`, `snn.py` | the LIF neurons, the CNN + LIF hybrid and the fully spiking network |
| `src/affect/models/cnn_transformer.py`, `heads.py` | the CNN + Transformer; the gender head |
| `src/affect/train/loop.py`, `data.py`, `settings.py` | one job: normalization, crops, batches, the training loop, early stopping, predictions |
| `src/affect/sweep/` | sweep expansion, the queue, the worker, smoke, status |
| `src/affect/analysis/validation.py`, `selection.py` | analyze; select and test-report |
| `src/affect/eval/metrics.py`, `stats.py` | UA, WA, F1; t intervals, the paired bootstrap, Holm |
| `src/affect/cli.py` | every command (`python -m affect.cli <command>`); `scripts/affect.sh <command>` runs it on the node with the environment |
| `scripts/` | node scripts: environment setup, checks, workers, the GPU schedule, packing results |
| `tests/` | 202 tests; `bash ~/affect/scripts/check.sh` runs them all on the node |
| `notebooks/cnn_3block.ipynb` | one experiment from its settings to its results |

## Commands used most

On the node, from any folder (`bash ~/affect/scripts/affect.sh <command> ...`):

- `smoke --sweep-file sweeps/<id>.json --expected-gpus 1` (with `CUDA_VISIBLE_DEVICES=<gpu>` in front)
- `enqueue --sweep-file sweeps/<id>.json`
- `status --sweep <id> --top 20`
- `analyze --sweep <id> --confidence 0.95 --n-boot 2000 --seed <any integer> --top 20`
- `select --sweep <id> --config <hash12> --name <record name> --reason "<one sentence>"`
- `test-report --selection <record name> --confidence 0.95`
- `requeue-failed --sweep <id>`; `requeue-running --all-workers-stopped`
- `compare-runs --sweep <id> --differences-shown 3` (checks that re-runs match earlier runs)

`scripts/smoke_enqueue.sh <id> [<id> ...]` smoke-tests and queues several sweeps in order, and stops a sweep whose smoke test fails.
