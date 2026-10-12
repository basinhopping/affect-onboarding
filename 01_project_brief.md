# F26 Affective Computing: project brief

Written 2026-10-12 for J. Carnegie's Claude project, from the records of the project so far. Read this first; the other files in this project go deeper.

## The project in one paragraph

Speech emotion recognition on IEMOCAP: a model hears one utterance (a sentence of acted dialogue) and names its emotion. The work is a research project for Dr. Sophina Luitel's class at Gonzaga (fall 2026). Harrison Christophersen built the code (the `affect` repository) and has run about 37,000 training runs on the department's GPU node since 2026-09-25. The best model so far is a tuned CNN on log-mel spectrograms, cnn_final_3, at 0.601 to 0.607 unweighted accuracy (UA) with test speakers never seen in training. The next step the project is considering is a pretrained speech model (HuBERT).

## People

| Who | Role | Email |
|---|---|---|
| Dr. Sophina Luitel | advisor; sets the questions and reviews results in meetings | |
| Harrison Christophersen | built the code and ran the experiments so far | hchristophersen@zagmail.gonzaga.edu |
| J. Carnegie | provided the IEMOCAP data (the raw release and a folder of Session 5 sentence wavs); joining the experiments | jcarnegie@zagmail.gonzaga.edu |

## The data

- IEMOCAP: 5 sessions, 10 actors (one man and one woman per session), 10,039 labeled utterances in the release.
- Labels: three external evaluators labeled every utterance; the released label is the category most of them named, when no other category was named as often (two of three for 5,149 utterances, all three for 2,383). The other 2,507 have no such category, are labeled xxx, and are left out.
- The standard set, used unless a result says otherwise: four classes, neutral (1,708), happy+excited (1,636: happy 595 and excited 1,041 merged), angry (1,103) and sad (1,084): 5,531 utterances.
- Other class sets the advisor asked for: five classes (the four plus frustrated, 1,849), four with frustrated in place of neutral, and three (happy+excited, angry, sad).
- 501 Session 5 sentence wav files are empty in the release. The code fills them from J. Carnegie's wav folder, so the training copy has all 10,039 utterances.
- Training reads one folder, `data/iemocap/assembled` (1.4 GB), built from the raw folders with a manifest of every file's SHA-256. Its manifest's own SHA-256, d6c6aefcc5de2b5471fd06e5f116f9e299ce2ed11b73ecfddbd7f2125fdaae0e, identifies the exact data every result used.

## How results are measured

- UA (unweighted accuracy): the recall of each class, averaged with equal weight. Chance is 0.25 with four classes. WA (weighted accuracy) is plain accuracy, which papers often report as "accuracy".
- Session out: five folds; each tests one session, validates on the next (session 5 on session 1) and trains on the other three. Test speakers are never in training. This is the main protocol.
- Pooled: all sessions mixed and dealt into five groups by speaker and class; the same speakers appear in training and test, so numbers are higher.
- Validation decides when training stops and which epoch to keep, and which settings to choose. Test files are read only after the choice is recorded (see 02_how_we_work.md).
- Every number is the mean over training seeds (usually 5) with a 95% interval; comparisons use a paired bootstrap and Holm's correction for multiple comparisons.

## Results so far (test UA)

| Model | Session out | Pooled |
|---|---|---|
| Baseline: 4-block CNN (32 to 256 channels), mean pooling | 0.553 (seeds 6 to 10); 0.559 (seeds 1 to 5) | 0.602 |
| cnn_final_3: the tuned 4-block CNN | 0.607 (seeds 6 to 10); 0.601 (seeds 1 to 5) | 0.678 |
| The advisor's 3-block CNN (32, 64, 128 channels, lr 0.001), as specified | 0.535 (seeds 6 to 10) | 0.585 |
| The same 3-block CNN after a sweep of its settings | 0.601 (seeds 6 to 10) | 0.654 |
| CNN + LIF hybrid (CNN front end, spiking LIF layers) | 0.595 (seeds 6 to 10) | 0.662 |
| Fully spiking CNN + LIF | 0.555 (seeds 6 to 10) | 0.615 |
| CNN + Transformer | 0.605 (seeds 6 to 10) | |
| SVM on openSMILE features (one run) | 0.621 | |

Class sets, cnn_final_3, session out: five classes 0.505, four without neutral 0.580, three classes 0.723. Pooled: 0.570, 0.641 and 0.783.

Published context: clean speaker-independent CNNs trained from scratch reach 56.6 to 61.8 UA; frozen pretrained speech models with a small classifier reach about 65 to 72 (wav2vec 2.0 Large 65.64, HuBERT Large 67.62, WavLM Large 70.62 accuracy in the SUPERB benchmark, four classes, leave one session out).

## Open questions and next steps

- Night 12 (2026-10-11 to 10-12) tested the pooling of the two best CNNs (test UA, session out, seeds 6 to 10): SD pooling helps the tuned 3-block CNN (0.601 against 0.576 with mean pooling) and makes no detectable difference in cnn_final_3 (0.608 against 0.607); attention pooling is worse in both. Every file is in examples/.
- Pretrained models: HuBERT is the likely next model; the meeting doc of 2026-10-12 lists the pretrained models other papers have used on IEMOCAP, with their protocols.
- Whether final test numbers should come from retraining on training plus validation sessions is still open (all results above train on three sessions).
