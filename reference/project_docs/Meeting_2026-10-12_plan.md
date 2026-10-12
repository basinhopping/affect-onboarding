# Meeting of Monday 2026-10-12: plan (written 2026-10-08 evening, Pacific; updated 2026-10-12)

The agenda is Dr. Luitel's list from the meeting of 2026-10-02 (Harrison's notes) and nothing else. Every item is shown on session out and on the pooled ("generalized") split only.

Docs: the meeting doc "Results for the meeting of October 12" (https://claude.ai/code/artifact/18d30230-e74d-4e2d-a1d2-be68a2aeaeae) and Harrison's private prep sheet "Meeting prep: likely questions (private)" (https://claude.ai/code/artifact/4beee553-1178-46ac-9a39-61cad6781260). Both hold night 10's confirmed results, night 11's pooled and class-set results, the GitHub link and the notebook, and since 2026-10-12 the pretrained models papers used on IEMOCAP and the search section's p-values from 1,000,000 draws. Night 12's pooling results went into both on 2026-10-12 at about 18:05 UTC (meeting doc revision 69).

Code: https://github.com/basinhopping/affect (pushed by Harrison on 2026-10-11, one commit d193f4e, 293 files: code, sweeps, tests, findings.csv and notebooks/cnn_3block.ipynb; no data, no run folders, no HANDOFF.md, VERIFICATION_LOG.md or node login). It was created public, where the plan said private; making it private, and then adding Dr. Luitel as a collaborator, is Harrison's call. Not on GitHub yet: CLAUDE.md, the two night-12 pooling sweeps and four first-run sweeps (lines in artifacts/analyses/onboarding_20261012/git_lines.txt; expected " 7 files changed, 634 insertions(+)").

## Her list

1. Confusion matrices and accuracy per class for the models, to see which class each model struggles with.
2. The highest CNN accuracy on IEMOCAP in published papers (CNNs specifically), and whether ours is low ("some CNNs have reached over 70%").
3. Her class sets: 5 classes (neutral, anger, happy+excited, sadness, frustration); 4 classes without neutral (anger, happy+excited, sadness, frustration); "a mix of other classes, we should see what works".
4. Other models that would work with this data, and why this CNN is low.
5. CNN + LIF.
6. A 3-level CNN: channels 32, 64, 128, lr 0.001, and a sweep around those settings.
7. What the grid searches were about, specifically pooling and SD pooling.
8. Training accuracy, and whether the model over- or underfits ("pretty key").
9. Code structure: the code on GitHub, and a notebook showing how each config is set up and run as jobs.

## Night 12 results (2026-10-12, 18:05 UTC)

Pulled at 17:55 UTC (200 of 200 jobs done, 0 failed; 1,648 files checked on the PC) and analyzed in the rule's order (artifacts/analyses/night12_20261012/README.txt): validation files only (1,208), analyze, a repeat check (the 50 runs of the two recorded configurations equal their earlier copies), the 8 selection records, then the 400 test files, test-report and families N12A and N12B. Test UA, session out, seeds 6 to 10:

- Tuned 3-block CNN: SD (its own) 0.601, max 0.594, mean 0.576, attention 0.564. Minus SD: mean -0.025 and attention -0.037 (Holm p 0.0015 each, worse); max -0.007 (Holm p 0.129, not detectable).
- cnn_final_3: SD 0.608, mean (its own) 0.607, max 0.600, attention 0.583. Minus mean: SD +0.001 (Holm p 0.777) and max -0.007 (Holm p 0.288), not detectable; attention -0.024 (Holm p 0.0015, worse).
- So SD pooling helps the tuned 3-block CNN and makes no detectable difference in cnn_final_3; attention pooling is worse in both. In the meeting doc: the paragraph "Tested directly on both finalists." in the 3-block section, and the last sentence of "Pooling in the grid".

## Status on 2026-10-12, 09:00 UTC (Monday, early morning Pacific)

Night 12, the pooling test of the two finalists (for item 7): two sweeps, 200 jobs, session out, four classes, seeds 6 to 10. cnn_3block_pooling_1 is the tuned 3-block CNN (threeblock_final, recorded with SD pooling) with mean, max and attention pooling; cnn_final_3_pooling_1 is cnn_final_3 (recorded with mean pooling) with SD, max and attention pooling. RULE_night12_tests.txt (analyses/night12_prep_20261012) was fixed at 06:27 UTC and amended at 06:44 UTC to 2,000 draws, before any file of night 12 left the node; families N12A and N12B, each variant minus its finalist's recorded pooling, Holm over 3. Harrison pushed the tree (TREE OK 5a49cdce..., 202 tests OK, check: all three passed) and started it Sunday night; at 23:44 PDT 12 jobs ran on the four GPUs and none had failed. Morning: lines/5_pack_night12.txt on the node, pull.ps1 on the PC, then analyses/night12_20261012 in the rule's order. Its dry run passes after two fixes made before any night-12 file reached the container (2,000 draws in the test analysis; a configuration with no complete run is listed as unfinished instead of stopping the selection script).

Meeting doc (revision 67) and prep sheet (revision 51), changed overnight:

- Other models: the table lists the pretrained models papers used on IEMOCAP (wav2vec 2.0, HuBERT, WavLM, UniSpeech-SAT, data2vec, emotion2vec, Whisper), with sizes, training (frozen or fine-tuned), results and cautions, and the HuBERT checkpoints. Frozen with a small classifier, session out: 63.4 to 72.9% accuracy; fine-tuned HuBERT 73.5% UA (Morais et al. 2022). Every figure was checked on its paper by a second agent; the full figures, protocols and checkpoints are in the project doc "claude/Pretrained models on IEMOCAP.md".
- How I searched the settings: the bootstrap p-values from 1,000,000 draws (analyses/weekly_20261012/doc_facts_bootstrap.txt), since 2,000 draws cannot show a p below 0.001. The 5x5 kernel alone: Holm-adjusted p 0.002 (still the only detectable single change; patience 20 alone 0.108). Grid 1's best cell 0.024, grid 2's 0.0005. Of the 3-block sweep's 100 random combinations, 34 are better and 6 worse (36 better with 2,000 draws); no single-change or grid verdict changed.
- A third agent checked the night's edits against the files and papers; its fixes were applied (for example, the SVM's per-speaker z-scores use the test speakers' unlabeled audio, as Pepino et al.'s do; the tuned 3-block CNN is "not detectably different" from cnn_final_3 rather than "level with" it).

Onboarding kit for a new collaborator: C:\Affective Computing\affect_onboarding (README_FIRST.md, eight numbered files, project_instructions.txt, iemocap_index.csv, sweeps/), CLAUDE.md at the repository's root, and four first-run sweeps (the baseline CNN under the four class sets, seed 1; expected test UA 0.5520, 0.4345, 0.5167, 0.6873). The steps sheet "Getting started with the affect code" and the email draft are ready; Harrison may go through it in person instead.

Tree digest now 53867f195fb79b0eab5e5d89aa95a10bcd19e9bbff10f583f3527b25d18b28b1, the same on the PC and in the container.

## Status on 2026-10-11, 23:00 UTC (Sunday, Pacific)

Night 11 ran the three finalists (tuned 3-block, CNN + LIF hybrid, fully spiking) under pooled folds and the three class sets: 21 sweeps, 525 jobs, 0 failed, Sat 18:12 to Sun 09:11 PDT on all four GPUs (178.6 worker-hours). Pulled and analyzed as RULE_night11_tests.txt fixed it (artifacts/analyses/night11_20261011/README.txt): 21 selection records before any test file, then the test reads, families N9E, N9F and N9G (pairing seed 20261010).

Test UA, seeds 1 to 5:

- Four classes, pooled: cnn_final_3 0.678, hybrid 0.662, tuned 3-block 0.654, spiking 0.615, baseline 0.602, her 3-block as specified 0.585.
- N9E: tuned 3-block pooled minus her version +0.069, minus the baseline +0.053, minus cnn_final_3 -0.023 (Holm p 0.0015 each).
- N9F: hybrid pooled minus the baseline +0.060 (Holm p 0.002), minus cnn_final_3 -0.016 (Holm p 0.006); spiking minus the baseline +0.013 (Holm p 0.09, not detectable), minus cnn_final_3 -0.063 (Holm p 0.002).
- N9G (class sets, both splits): the tuned 3-block and the hybrid beat the baseline in all 12 comparisons (+0.029 to +0.049, Holm p 0.009 each); the spiking network in none (-0.001 to +0.016).
- Under every set and split cnn_final_3 is first (0.011 to 0.030 ahead of the tuned 3-block and the hybrid, not tested) and her 3-block as specified last. Three classes: cnn_final_3 0.723 session out, 0.783 pooled; tuned 3-block 0.712 and 0.771; hybrid 0.712 and 0.768.
- Training gap at the kept epoch, pooled and class sets: the hybrid's is the largest in every setting (0.095 to 0.160); spiking 0.065 to 0.102 and tuned 3-block 0.038 to 0.079 overlap (night11_trainacc.txt).

The meeting doc's lead chart shows the four-class results on both splits (session out on seeds 6 to 10 from night 10, pooled on seeds 1 to 5), and the class-sets section has a chart of all six models under the three sets and both splits. A second agent checked every new number against the files (all correct); its wording points were fixed (the lead is scoped to the six CNN models, since the one-run SVM scored 0.621; Setup says differences are computed before rounding). The prep sheet carries the same results.

GitHub preparation (artifacts/analyses/github_prep_20261011/README.txt): the node login moved to node.local.txt (PC only, ignored by git); a trial git add on a copy caught that the unanchored .gitignore pattern data/ also hid src/affect/data/, fixed by anchoring every top-level pattern. Tree digest after the push record in HANDOFF.md: 50ef1156... (the same on the PC and in the container).

## Status on 2026-10-10, 19:40 UTC (Saturday, Pacific)

Night 10 (the three confirmations, 1,000 jobs, 0 failed) was pulled and analyzed (artifacts/analyses/night10_20261010/README.txt). Before the pull, RULE_night11_tests.txt (artifacts/analyses/night11_prep_20261010, saved 18:49 UTC) fixed night 11's run and its tests.

Test UA, session out, seeds 6 to 10 (families N9B and N9C of RULE_night9_tests.txt, pairing seed 20261009):

- Her 3-block CNN as specified 0.535; after the sweep (threeblock_final, e0c211fcced3) 0.601: +0.066 over her version and +0.048 over the baseline (Holm p 0.0015 each), -0.006 against cnn_final_3 (p 0.245, not detectable). Changes: channels 64, 128, 256; 5x5 kernels; SD pooling; a 128-unit hidden layer; dropout 0.2; batch 16; 10 s crops; 40 mel bands; SpecAugment; up to 100 epochs, patience 20; lr 0.001 unchanged.
- CNN + LIF hybrid (lif_final, 553e54c97626) 0.595: +0.041 over the baseline (Holm p 0.002), -0.013 against cnn_final_3 (p 0.036, Holm p 0.072, not detectable). Fully spiking (snn_final, 396d6ef8c242) 0.555: +0.001 against the baseline (p 0.93), -0.052 against cnn_final_3 (Holm p 0.002).
- Beside them: baseline 0.553, cnn_final_3 0.607. Her 3-block as specified is again below the baseline (-0.019, p 0.007).
- Training gap at the kept epoch: her 3-block +0.020 (underfits), tuned 3-block +0.077, hybrid +0.149, spiking +0.106 (night10_trainacc.txt).

Earlier results (night 9, seeds 1 to 5) stand as written in the meeting doc: baseline 0.559 and 0.602 pooled; cnn_final_3 0.601 and 0.678; class sets; literature; training accuracy.

## Schedule

- Sat 10-10 evening: Harrison pushed the tree (digest 7207d264...), installed and checked it, and started night 11 at 18:05 PDT; at 18:17 he gave it all four GPUs until noon Sunday. The queue emptied at 09:11 Sunday.
- Sun 10-11: pulled, recorded, test reads, families N9E, N9F and N9G; the meeting doc and prep sheet filled and checked; the code on GitHub with the notebook. Sunday night: night 12 prepared, pushed (digest 5a49cdce...) and started.
- Mon 10-12 morning: night 12's pack (line 5) and pull, then its analysis (analyses/night12_20261012) and the pooling results in both docs. Afternoon: the meeting. The next push to the node carries tree digest 53867f19...

## Meeting format

One short doc with one section per item: the answer first, then one table or figure. Not included: CNN configuration history, the 27-split study, grid-search details beyond the pooling answer. Anything off the list can be sent afterwards.
