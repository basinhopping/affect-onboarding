# Start here: the affect project kit for J. Carnegie

This kit gives a Claude project everything it needs to work on the affect project from the first message: what the project is, how it works, the code, the setup from your raw IEMOCAP folder, a first short run with its expected results, a real night as an example, and how much has been run so far. Harrison will go through it with you.

## The files

| File | What it is |
|---|---|
| 01_project_brief.md | the project, the people, the data, how results are measured, the results so far |
| 02_how_we_work.md | the working rules (the node, the GPUs, test files, records) and why |
| 03_repository_guide.md | the code: one experiment start to finish, a sweep file line by line, run folders, where things are |
| 04_setup_from_raw_data.md | from your raw folder to a checked install on the node, with the hash your data must match |
| 05_first_run.md | the first run: the baseline CNN on the four class sets, seed 1, 20 jobs, every line and the expected results |
| 06_example_night.md | a real night from Harrison's runs: the lines, the output, the morning analysis |
| 07_scale_of_the_work.md | every run so far: 37,335 runs, 1.8 million epochs |
| 08_starting_plan.md | a first plan, step by step |
| project_instructions.txt | the text for the project's instructions (a Claude project) |
| CLAUDE.md | the same instructions, which Cowork reads from this folder |
| examples/ | a real night (night 12) with every file: the sweep maker, the RULE, the exact lines, the morning analysis and its outputs |
| reference/ | the full background: the meeting doc of 2026-10-12, a line-by-line code walkthrough, the project's plan, literature review, split study and pretrained models, and Harrison's working instructions for Claude |
| phase0_audit/iemocap_index.csv | the utterance index the data build checks against (needed for 04) |
| sweeps/ | the four first-run sweep files, in case your clone does not have them |

## Setting up Claude

1. Make a Claude account if you do not have one, and create a project named "F26 Affective Computing". Free accounts can create up to five projects ([Claude Help Center](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects)).
2. In the project, add the eight numbered .md files to the project knowledge, and paste the text of `project_instructions.txt` into the project instructions. Leave `phase0_audit/` and `sweeps/` out of the project knowledge; they are data, not reading. Add files from `reference/` too if the project has room; the numbered files are enough to start.
3. Start a chat in the project with: "Read the project files. I am at step 1 of 08_starting_plan.md; walk me through it one command at a time."
4. At the end of each working session, ask Claude for a state note, save it as `state_<date>.md`, and add it to the project knowledge in place of the previous one. The project's files are what every chat in it reads, so the state note is how the next chat knows where you stopped.

With the free plan, Claude works in the chat: it cannot open the files on your computer or run commands there, so you run each command and paste the output back (or attach a file to the chat). Cowork, where Claude works directly in a folder on your computer (building the data, editing sweep files), needs a paid plan: Pro, Max, Team or Enterprise ([Get started with Claude Cowork](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork)). With Cowork, connect this kit's folder (it holds your raw IEMOCAP folders too); Claude reads `CLAUDE.md` here, and the affect repository has its own `CLAUDE.md` once its newest version is on GitHub. In either case, commands on the GPU node are yours to run: Claude cannot log in to it.

## What you need from Harrison

- Access to this repository (it is private; Harrison invites you on GitHub).
- A word before each night you plan to use the GPUs, so that your jobs and his do not share a GPU.
- If the data build gives trouble: his copy of `data/iemocap/assembled`.
