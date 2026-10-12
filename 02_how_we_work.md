# How this project works: rules for J. Carnegie and his Claude

These are the working rules the project has followed since its first run. They exist because the GPU node is shared, because results go to an advisor, and because a mistake in a test result cannot be taken back once it has been seen. Claude should follow them without being asked, and say so when a request would break one.

## Who does what

- Claude prepares everything: sweep files, scripts, exact commands, analyses and write-ups. It checks each one before handing it over.
- The person runs every command on the GPU node (Claude cannot log in to it) and pastes the output back. Claude reads the output, says whether it is what was expected, and gives the next line.
- One step at a time, each with what it should print. A line that prints something unexpected stops the sequence until it is understood.

## The GPU node: seas-gpu-node-01

- Log in with your own account (`ssh jcarnegie@seas-gpu-node-01`, if that is your username there). Everything lives in your home folder: the code in `~/affect`, the Python environment in `~/venvs/affect`, model downloads in `~/cache`. Nothing in this project uses `/scratch`.
- The node has 4 GPUs and other users. On weekdays, GPU jobs run in the evening and overnight only, and every worker stops by morning: the start line sets a stop time (8:00 or 10:00 Pacific) that writes `~/affect/queue/STOP`.
- Before starting anything, check `nvidia-smi` and use only GPUs no one else is on. Tell Harrison before a run, so that two people's jobs do not land on the same GPU.
- Work runs in tmux sessions that end by themselves (`gpu0` to `gpu3` for workers, one named session for a night's schedule). Never start anything that restarts itself or runs forever in the background. Never touch a tmux session you did not start.
- To stop early: `touch ~/affect/queue/STOP`. Each worker finishes its current job and exits. Remove the file (`rm -f ~/affect/queue/STOP`) before starting workers again, because a worker exits at once while it exists.
- Never change the code in `~/affect` while workers run: a running worker loads some files later, and would mix old and new code. Stop the workers, wait until `tmux ls` lists no gpu session, then update.

## Results: what makes a number trustworthy

- Every number comes from a file the code wrote, and is rounded half up from the exact value (0.6006 shows as 0.601). Nothing is estimated or remembered.
- Test, never assume: a new sweep file is checked with the repository's own sweep expansion and smoke-tested before it is queued; a new command is tried on stand-in data first; files copied between machines are checked by SHA-256.
- Choices use validation only. `analyze` compares configurations on validation files; `select` records the choice; only then does `test-report` read that configuration's test files. A test number is never used to choose anything.
- Before results come back, write down which test files will be read and which comparisons will be made (a RULE file with the time it was fixed). Once a test file has been read, the rule does not change. This is what makes a p-value mean what it says.
- Comparisons use a paired bootstrap over seeds and test utterances, with Holm's correction within each family of comparisons. Report the difference, its interval and the Holm-adjusted p; a difference that is not detectable is reported as such, not as "slightly better".
- Five seeds for a result, three for screening many configurations, one only for learning or smoke tests.

## Records

- Each night has a prep folder (`artifacts/analyses/nightN_prep_<date>/`: the sweep-making script, the RULE file, the lines, a README) and an analysis folder (`artifacts/analyses/nightN_<date>/`: scripts, outputs, a README with what was run and what it found).
- Run folders are never overwritten or deleted. A failed job's folder is kept as evidence.
- The repository on GitHub (github.com/basinhopping/affect) is public: no logins, host names, passwords or data ever go into it. IEMOCAP is licensed, so `data/` and the results in `artifacts/` stay out of git.

## Writing for the advisor

- Documents Dr. Luitel reads are written in the first person ("I"), in plain prose, with every number traceable to a file. No dashes as punctuation.
- Say what a result shows and what it does not; if something was not tested, say so.
