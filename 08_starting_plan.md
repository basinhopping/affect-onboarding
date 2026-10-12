# A starting plan

A first plan for J. Carnegie, to be adjusted with Harrison and Dr. Luitel. Each step ends with something checked.

## Step 1: set up (with Harrison, in person)

- Claude: the project in Claude with these files as its knowledge and the instructions in `project_instructions.txt` (README_FIRST.md says how).
- Data: build `data/iemocap/assembled` from your raw folders (04_setup_from_raw_data.md, part 1). Done when assemble reports manifest d6c6aefc... and verify-data says `"ok": true`.
- Node: clone, environment, data copied, `check.sh` (parts 2 and 3). Done when it prints `check: all three passed`.

## Step 2: the first run (one evening)

- 05_first_run.md: the baseline CNN under the four class sets, seed 1, 20 jobs. Done when the four test reports print 0.5520, 0.4345, 0.5167 and 0.6873, or Claude has explained any difference (usually package versions).

## Step 3: learn the code

- Read 03_repository_guide.md and the notebook `notebooks/cnn_3block.ipynb` on GitHub.
- Ask Claude to walk through `src/affect/models/cnn.py` and `src/affect/train/loop.py` line by line, using your first run's folders as the example (log.txt, metrics_val.json, efficiency.json). Harrison also has a line-by-line walkthrough of the whole repository he can share.

## Step 4: a first real comparison

- Pick one change to test (for example a 5 by 5 kernel, or SD pooling, on the baseline under one class set). Ask Claude to write the sweep (base plus the change, five seeds) and a RULE file saying which comparison will be read on test, before anything runs.
- Run it in one evening (50 jobs), analyze on validation, record the selection, then read the test comparison.
- Done when the result is written up in a short README with its numbers and where they come from.

## Step 5: a direction

The project's likely next step is a pretrained speech model (HuBERT; the meeting doc of 2026-10-12 lists what other papers used). Agree with Harrison and Dr. Luitel on what you take on, then plan it with Claude the same way: what question, which runs, which comparisons, before anything runs.

## Working with Claude, day to day

- Start a chat in the project and say what you want to do; Claude knows the project from these files.
- Claude prepares the files and the exact lines; you run them on the node and paste the output back.
- Ask why whenever a step is not clear; the rules in 02_how_we_work.md each have a reason.
