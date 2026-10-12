# affect onboarding kit (private)

Keep this repository private: `phase0_audit/iemocap_index.csv` comes from IEMOCAP's licensed annotations, and the files name the GPU node.

This kit gets you from your raw IEMOCAP folder to a first run on the GPU node, with Claude guiding each step. The code itself is the public repository https://github.com/basinhopping/affect; this kit does not copy it.

## Start here

1. **Get this folder.** On this page, Code, Download ZIP. Unzip it into your home folder and rename the folder `affect_onboarding` (or `git clone` this repository there).
2. **Drop in the raw data.** Put your `IEMOCAP_full_release` folder and your `wav` folder (the Session 5 sentence wavs, 31 dialog folders) inside `affect_onboarding`. Git ignores both.
3. **Set up Claude.** With Cowork: connect the `affect_onboarding` folder and say "Read CLAUDE.md, then take me through 08_starting_plan.md one step at a time." With a Claude project instead: add the eight numbered files to the project's knowledge, paste `project_instructions.txt` into its instructions, and start a chat with the same sentence. README_FIRST.md has the details.
4. **Build the data** on your computer: 04_setup_from_raw_data.md, part 1. It must report manifest d6c6aefc... and `"ok": true`.
5. **Set up the node** under your own login: 04, parts 2 and 3. It must print `check: all three passed`.
6. **First run:** 05_first_run.md, 20 short jobs on one GPU, in the evening. The test UA should read 0.5520, 0.4345, 0.5167 and 0.6873.

Claude gives you every command with what it should print; you run it and paste the output back.

## Layout

```
affect_onboarding/
  README.md, README_FIRST.md, CLAUDE.md, 01 to 08, project_instructions.txt    the kit
  sweeps/                          the four first-run sweep files
  phase0_audit/iemocap_index.csv   the utterance index the data build checks against
  examples/                        a real night (night 12): every file, from the RULE to the results
  reference/                       the project's full background (README.md there lists it)
  IEMOCAP_full_release/            yours: the IEMOCAP release
  wav/                             yours: the Session 5 sentence wavs
```
