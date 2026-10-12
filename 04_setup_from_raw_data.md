# Setting up: from the raw IEMOCAP folder to a checked install on the node

Three parts: build the training data on your computer from your raw folders, set up the code under your own login on the node, and check the install. The data commands were run on a fresh copy of the code with a small stand-in copy of IEMOCAP laid out as this kit is, before this was written; the node commands are the ones Harrison's own setup used. Paste each step's output to Claude before going on.

## 1. Build the data on your computer

Training reads one folder, `data/iemocap/assembled`: the 10,039 utterance wav files plus the label and transcript files, 1.4 GB in all. The code builds it from three things:

| What | Where it comes from | What it is for |
|---|---|---|
| `IEMOCAP_full_release` | your copy of the release | the audio, labels and transcripts |
| `wav` | your folder of Session 5 sentence wavs (31 dialog folders) | 501 Session 5 files are empty in the release; the assembly takes those from here |
| `iemocap_index.csv` | this kit's `phase0_audit/` (4.3 MB; SHA-256 feed11b09d82b8d2b09baff06a0460f049ee4b3fe588001997edffaa00534f45) | the index of all 10,039 utterances, from Harrison's first audit of the data, that the assembly checks against |

This kit is that folder: put your `IEMOCAP_full_release` and `wav` folders inside it, next to `phase0_audit/` (README.md shows the layout). The commands assume the kit is at `~/affect_onboarding`; if it is somewhere else, change the first line. You need git and Python 3.10 or newer with numpy (`python3 -m pip install numpy`). The commands are for a bash shell (macOS, Linux, or Git Bash or WSL on Windows; use `python` where your Python is not called `python3`):

```bash
K="$HOME/affect_onboarding"
cd "$HOME" && git clone https://github.com/basinhopping/affect.git && cd affect
PYTHONPATH=src python3 -m affect.cli ingest --dataset iemocap --name collaborator_wav --source "$K/wav" --max-seconds 3600
PYTHONPATH=src python3 -m affect.cli assemble --dataset iemocap --source IEMOCAP_full_release="$K/IEMOCAP_full_release" --source collaborator_wav="$K/wav" --source phase0_audit="$K/phase0_audit" --max-seconds 3600
PYTHONPATH=src python3 -m affect.cli verify-data --dataset iemocap --state first --max-seconds 3600
```

Each command prints its progress as JSON and continues where it stopped when run again.

- `ingest` copies the wav folder into `data/iemocap/raw/collaborator_wav` with the SHA-256 of every file. Run it until it prints `"phase": "finalized"`.
- `assemble` stops once after listing and copying, so run the same command again until it prints `"phase": "finalized"`. Your copy is byte for byte the same as Harrison's if it reports `"manifest_sha256": "d6c6aefcc5de2b5471fd06e5f116f9e299ce2ed11b73ecfddbd7f2125fdaae0e"`, with 10,342 files and 1,440,743,852 bytes. A different hash means some file differs from the copy every result so far used: stop and compare before training on it.
- `verify-data` re-reads every file; it also stops once (exit code 3), so run it again with the same `--state` until it prints `"phase": "done"` and `"ok": true`.
- `PYTHONPATH=src python3 -m affect.cli data-summary --dataset iemocap` then prints the class counts per session (four classes: 1,085, 1,023, 1,151, 1,031 and 1,241 utterances in sessions 1 to 5).

If this step gives trouble, Harrison can hand over his copy of `data/iemocap/assembled` instead; part 2 copies and checks it the same way.

## 2. Set up on the node

Log in with your own account (`ssh jcarnegie@seas-gpu-node-01`, if that is your username there). Everything goes in your home folder and touches no one else's files:

```bash
git clone https://github.com/basinhopping/affect.git ~/affect
bash ~/affect/scripts/setup_env.sh
```

`setup_env.sh` builds the Python environment once, in `~/venvs/affect` (about 6 GB, several minutes), and writes the exact package versions to `~/affect/artifacts/env/`. If the node cannot reach GitHub, copy your clone from your computer instead (`scp -r affect jcarnegie@seas-gpu-node-01:~/` from the folder above it); that brings the data along, so skip the two copy lines below.

Then, from the `affect` folder on your computer, copy the data across:

```bash
ssh jcarnegie@seas-gpu-node-01 "mkdir -p ~/affect/data/iemocap"
scp -r data/iemocap/assembled jcarnegie@seas-gpu-node-01:~/affect/data/iemocap/
```

## 3. Check the install

On the node, when `nvidia-smi` shows the four GPUs idle:

```bash
bash ~/affect/scripts/check.sh
```

It runs all 202 tests, re-hashes the 10,342 data files against their manifest (which catches anything the copy damaged), and checks the packages and each of the four GPUs with one training step. It must end with `check: all three passed`.

## Exact reproduction (optional)

Harrison's runs used torch 2.14.0, numpy 2.5.3, scipy 1.18.1, pandas 3.0.6, soundfile 0.14.0 and librosa 1.0.0 (every run's `provenance.json` lists them). `setup_env.sh` installs the newest versions, so if any differ, a re-run of one of Harrison's configurations can differ in the last digits. To match his exactly (not tried yet on a second account; check the versions in your first run's `provenance.json` either way):

```bash
~/venvs/affect/bin/python -m pip install --no-cache-dir torch==2.14.0 numpy==2.5.3 scipy==1.18.1 pandas==3.0.6 soundfile==0.14.0 librosa==1.0.0
```

Run it before the first sweep, then `bash ~/affect/scripts/check.sh` again.
