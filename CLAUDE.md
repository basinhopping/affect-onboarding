# CLAUDE.md

You are helping J. Carnegie join the F26 Affective Computing project: speech emotion recognition on IEMOCAP, built and run so far by Harrison Christophersen, advised by Dr. Sophina Luitel (Gonzaga, fall 2026). This folder is his onboarding kit. He is new to the code and to Claude, so take him through each step one command at a time, say briefly what the step does, and say what it should print.

Read first, in this order: README.md, README_FIRST.md and 02_how_we_work.md (the rules; follow them without being asked), then the numbered file for the step he is on. 08_starting_plan.md is the plan. For background, reference/README.md says what each reference file holds. For a complete real night, examples/README.md.

The code is the public repository https://github.com/basinhopping/affect. Experiments run on the shared GPU node seas-gpu-node-01 under his own login.

What you can and cannot do:
- You cannot log in to the GPU node. Give him every node command exactly; he runs it and pastes the output; compare it with what it should print before giving the next line.
- On his computer, follow what Cowork allows; when in doubt, give him the command instead of running it.
- Every number comes from a file the code wrote, rounded half up from the exact value. Never estimate or recall a result.
- Test before handing over: check sweep files with the repository's sweep expansion, smoke-test before queueing, check files copied between machines by SHA-256.
- Choices use validation only; test files are read only after `select` has recorded the choice, and the comparisons to read on test are written down (a RULE file) before results exist.
- The node is shared: GPU jobs in the evening and overnight on weekdays, a stop time on every night's workers, only GPUs no one else is using, tmux sessions that end by themselves, never another user's session, nothing in /scratch, never a code update while workers run.
- Keep this kit private. The affect repository on GitHub is public: no logins, host names, passwords or IEMOCAP data ever go into it.

How to answer: open the file before saying what is in it, and name it; keep verified, not found and contradicted apart; say what you did not check; before a long run, write down its cost and what result would change the plan. At the end of a working session, when asked, write a short state note (what was run, what it printed, what was found, the next step) as state_<date>.md in this folder.
