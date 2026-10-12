# IEMOCAP protocols and split plan

Written 2026-09-26 after the first meeting with Dr. Luitel; literature claims verified against the papers the same night; night-2 results added the next day. Holds the literature findings on how IEMOCAP is split, the split designs to compare, the rules for sweeping across splits, and the dataset facts that settled the "full dataset" question. The code plan lives in the repo (HANDOFF.md); this page is the reasoning. The full literature review, with every paper's protocol, preprocessing, model and results, is claude/IEMOCAP literature review.md. The night-2 results are in the Claude doc "CNN Results on IEMOCAP, Night 2".

Verification marks: [checked] means I re-checked the claim in the paper's own text. [reader] means a reader confirmed it on the page image while reading the whole paper, and I did not re-check it. Harrison pulled the papers into C:\Affective Computing\papers\; per-paper notes are in C:\Affective Computing\papers\notes\.

## 1. Dataset facts (from the Phase 0 audit files, _phase0_audit/README.md)

- Harrison's IEMOCAP_full_release is the complete corpus: 10,039 labeled utterances, 151 dialogs, 10 speakers, 5 sessions. Labels over all utterances: xxx (no majority) 2,507; fru 1,849; neu 1,708; ang 1,103; sad 1,084; exc 1,041; hap 595; sur 107; fea 40; oth 3; dis 2.
- Busso et al. 2008 give "ten thousand and thirty nine turns" (5,255 scripted) and approximately twelve hours [checked].
- The only defect: 501 empty Session 5 sentence wavs, filled from the collaborator's copy, each verified as a sample-exact cut of its dialog recording.
- The collaborator's spectrograms (spectrogram_cleaned/) are exactly the 5,531 utterances of neutral, happy, excited, angry, sad: "5 classes" with happy and excited apart, the standard 4 when merged. This is the "shortened" set. His Session 5 images are shifted: 1,228 of 1,241 show another utterance's audio, 645 of those with a different emotion label. Sessions 1 to 4 match.
- Class sets counted from the index (_phase0_audit/iemocap_index.csv): 4 merged (neu, hap+exc, ang, sad) 5,531; 4 without excited 4,490; 5 (hap and exc apart) 5,531; 6 (adds fru, the conversation-recognition set) 7,380; 7 (adds sur) 7,487. Decided with Harrison: classes with too little data are not added, and xxx (no agreement) is never used.
- Gender, 4 merged classes: female 2,649 (neu 733, hap+exc 770, ang 589, sad 557), male 2,882 (neu 975, hap+exc 866, ang 514, sad 527). Per session and gender: S1 F 528 M 557; S2 F 481 M 542; S3 F 522 M 629; S4 F 528 M 503; S5 F 590 M 651.

## 2. What the literature says about splits

- Two speaker-independent protocols dominate: leave-one-session-out (5 folds, ours) and leave-one-speaker-out (10 folds). Antoniou, Katsamanis, Giannakopoulos, Narayanan, ICASSP 2023 [checked]: recommend the 4 merged classes (5,531) and "a 10-fold Speaker Independent cross-validation with one speaker as test, eight as training and one as validation set". They call 5-fold and 10-fold equally reasonable and say 10-fold scores slightly higher, without testing it, and do not say how the validation speaker is chosen. Their own reproductions report WA only.
- Pepino, Riera, Ferrer, Interspeech 2021 [checked]: 5-fold leave-one-session-out with "one of the 4 training sessions to perform early stopping", i.e. our 3/1/1 protocol. Frozen wav2vec 2.0 Base (pretrained only, all layers weighted) reaches UA 67.2 +- 0.7 over 5 seeds with speaker normalization and 65.8 with global normalization. Their speaker normalization uses the test speaker's own audio statistics.
- Speaker overlap inflates results by about 6.5 WA points on the same model [checked]: Wang et al. 2021, partly fine-tuned HuBERT Large, 79.58 on random splits vs 73.01 speaker independent (10 folds); Gat et al. 2022, HuBERT Large, 78.3 vs 71.9 (5 folds of 2 test speakers). Using the test fold for validation adds 2.69 (session out) and 4.70 (speaker out) WA in emotion2vec's own table [checked].
- Script overlap: in Busso 2008 all ten actors "were asked to perform three selected scripts" [checked]. Pepino et al. 2020: holding the test script out of training drops their text model from about 0.87 to 0.79 AvAUC while the audio model stays at about 0.785 [reader, read from a chart]. Gamage 2018: word-based features score 64.0 UA on all test data and 52.2 on improvised-only test data [checked].
- Session 5 as a fixed test set is standard in conversation-level work (DialogueGCN, DialogueRNN) with 6 classes: 5,810 training and validation utterances, 1,623 test [checked]. DialogueGCN validates on 10% of the training dialogues [reader].
- Validation practice: of the 35 papers read that run their own IEMOCAP experiments, 8 let test data choose epochs or settings for at least one reported number, 14 describe no validation set or do not say how it was used, and 13 keep a validation set apart from the test data. 9 report spread over repeated runs.
- Gender: none of the 62 papers read trains on one gender and tests on the other. Vogt and Andre 2006 (German corpora, Naive Bayes): routing by the speaker's true gender added 4.3 UA (Berlin) and 3.6 (SmartKom) [reader]; with automatic gender detection, about 90% accurate, the overall gain was 2 to 4% [checked]. Her gender splits would be new results.
- Reference results, speaker independent, 4 merged classes [checked]: CNN-type models from scratch 56.6 to 61.8 UA (Satt 2017, Latif 2019, Aldeneh 2017); recurrent or attention models with heavy augmentation 64 to 68 (Latif 2022, 2023); classical features with an SVM 56.8 to 58.8 (Gamage 2018, Latif 2019); frozen wav2vec 2.0 Base under our protocol 67.2; fine-tuned self-supervised models 69.9 to 77.8 (Chen and Rudnicky 2023, Morais 2022). The earlier ranges on this page (CNNs 57.6 to 61.8, SVMs 58.7 to 61.7) were secondhand and are corrected here.

## 3. Splits to compare (Dr. Luitel's ideas plus the leakage ladder)

Dr. Luitel offered these split ideas as possibilities to look at, not decisions (Harrison, 2026-09-26). The aim is one baseline-CNN row per option, with 5 seeds and intervals, so the whole spread can be discussed at the next meeting.

| Split | Train / validation / test | What overlaps between train and test |
|---|---|---|
| A. Session out (current) | 3 sessions / next session / 1 session, 5 folds | script lines only |
| B. Speaker out, literature default | 8 speakers / the test speaker's session partner / 1 speaker, 10 folds | script lines only; 4 sessions of training |
| C. Speaker out, partner in training | 8 speakers including the partner / a speaker of another session / 1 speaker, 10 folds | the other half of the test speaker's conversations |
| D. Conversation grouped | dialogs in 5 groups; next group validates | speakers |
| E. Pooled random ("all together") | utterances in 5 stratified groups; next group validates | speakers and conversations |
| F. Within session | 5 stratified groups inside each session (25 folds); also train on one actor, test on the partner (10 folds) | one recording setup |

Leakage ladder: E, D, C, A. Each rung removes one overlap (speaker, conversation, session); same model, same seeds.

Gender, built on the session-out folds: a male-only model (train and validation on male speakers) and a female-only model, each tested on the whole held-out session, so one run gives M to M, M to F and M to all. Gender-routed system ("user checks a box"): male model for male test speakers, female model for female ones, on the same test utterances as A. Half-data control: a mixed model trained on a stratified half of A's training data, separating "specialized by gender" from "trained on half the data". Caveat: five people per gender.

Content: improvised only, scripted only, and train on improvised then test on everything (the script check).

All of the above ran as baselines on night 2 (results in section 8). Candidates from the reading for a later run (literature review, section 12): session out with 4 training sessions and a random 20% validation (the pretrained-model convention); the test session as validation, as a labeled measurement of selection bias; Session 5 as test (the conversation convention), with 4 and 6 classes; a training-size curve at 25%, 50% and 75%; a script-disjoint test (Pepino 2020); within-speaker folds; and cross-corpus once a second dataset is in.

## 4. Rules

1. Tune on one protocol: session out, 4 merged classes. Night 2 supports this: rankings of 41 configurations correlate only 0.33 (pooled random) and 0.39 (female-only training) with session out.
2. Every other split evaluates frozen finalists (baseline plus the top 3 after the confirmation), 5 seeds, test read once per split.
3. Whether tuning transfers is tested: one 3-seed sweep of about 40 configurations under two contrasting splits (pooled random, gender specific), compared by rank correlation with session out.
4. Validation always comes from the training side.
5. Each split is defined once in code with a name and a hash; every run records it. One protocol per sweep.
6. Speaker-dependent results get their own labeled table.
7. Confirmation rounds use seeds never used in screening (night 2's confirm round reused screening seed 1 for 1 of 5 seeds).

## 5. Her other points

- TPOT evolves scikit-learn pipelines with genetic programming (population 100, 100 generations in the original paper) and has no neural network operators; TPOT-NN adds only a logistic regression and a multilayer perceptron [checked]. So TPOT cannot tune the CNN, Transformer or spiking models. It fits a classical AutoML baseline on openSMILE features (eGeMAPS, ComParE), where published SVM baselines sit at 56.8 to 58.8 UA. Not run yet: it needs openSMILE features and its own environment on the node.
- Grid search: our 18 settings at the screened levels multiply to about 90 million combinations, so a full grid only on the 3 or 4 settings that matter most, 3 seeds. Bergstra and Bengio 2012 support this: "Random searches of 8 trials match or outperform grid searches of (on average) 100 trials" [checked]. The first grid, cnn_grid_1 (n_mels 40 or 64, SpecAugment off or mild, class balance none or weighted loss, mean or stats pooling: 16 combinations, 240 jobs), ran on night 2: mild SpecAugment +1.0 points of validation UA, 40 mel bands +0.7, statistics pooling -0.8, weighted loss 0.
- Batch normalization: every block uses it now. Group normalization only beats it at 8 or fewer examples per batch (ImageNet error at 32, 16 and 8 per GPU: 23.6, 23.7, 24.8 for batch norm vs 24.1, 24.2, 24.0 for group norm) [checked]; our batches are 16 to 32. SubSpectral normalization (batch norm per frequency band) gained about 1 point on keyword spotting and acoustic scenes [reader]. None of the 62 papers compares normalization layers for emotion recognition. Plan: make the normalization layer a setting (batch, subspectral with 2 or 4 bands, group, none).

## 6. EmoFilm (parked by Harrison on 2026-09-26: one dataset at a time)

Files checked in C:\Affective Computing\EmoFilm (readme, legend spreadsheet, every wav header):
- 1,115 clips, flat folder, readme and f_m_corpus_it_es_en_legend.xlsx (file, speaker, film for every clip).
- Languages: Italian 413, Spanish 359, English 343 (English is original film audio, the others dubbed).
- Emotion codes: tri sadness 254, gio happiness 240, rab anger 232, ans fear 221, dis 168. The paper (Parada-Cabaleiro et al. 2018) settles "dis": the fifth class is contempt, and the authors excluded disgust [checked]. No neutral class.
- Mono, 16-bit, 44.1 kHz (IEMOCAP is 16 kHz, so resample into a separate verified copy).
- Clips median 1.87 s, max 5.00 s, 38.4 minutes in total. The paper states a mean of 3.5 s (SD 1.2), which our files contradict [reader].
- 591 scenes (gender, emotion, number, take); 163 in all three languages, 198 in two, 230 in one. 215 named speakers plus 17 unnamed clips; 54 speakers appear in more than one film. The paper gives 207 speakers and 43 films [reader]. Splits must keep a scene's language versions together and group by speaker.
- Shared classes with IEMOCAP: anger, happiness, sadness.
- The paper prints English 360 and Spanish 342, but its own per-emotion averages give 343 and 359, our file counts [reader].

## 7. Reporting (agreed 2026-09-26)

- After every run, a report in the format of "First CNN Results on IEMOCAP": early findings first, then what I did, then a "How I found this" process section that answers "how did you get this" for each result, then details. First person ("I").
- Each report compares against the previous best ("last week's best was x; this week y beat it by z").
- "How it works" becomes its own standing page.
- Before each meeting, one weekly summary of the week's reports; slides from it when needed.

## 8. Status (updated 2026-09-26, 19:45 UTC)

- Night 2 ran 7,010 jobs from 7:17 pm to 10:26 am Pacific with 0 failures. Results, from the Claude doc "CNN Results on IEMOCAP, Night 2" (test UA, 5 seeds, 95% intervals):
  - Confirmed winner cnn_final_1 (mild SpecAugment, class-weighted loss, max pooling, 10 s crops, batch 16, learning rate 0.0003, label smoothing 0.1): 0.580 [0.561, 0.600] against the baseline's 0.559 [0.549, 0.569]; paired gain +0.021 [+0.004, +0.039].
  - Leakage ladder: pooled random 0.602; within session 0.566; conversation grouped 0.565; speaker out 0.564; session out 0.559; speaker out with partner in training 0.558; cross-speaker within a session 0.435.
  - Gender: routed by gender 0.531 vs mixed 0.559 (-0.028 [-0.041, -0.014]); ties a half-data mixed model (+0.005 [-0.006, +0.017]). M to M 0.532, M to F 0.469, F to F 0.534, F to M 0.431.
  - Class sets: 0.559 (4), 0.529 (4 without excited), 0.453 (5), 0.368 (6), 0.320 (7), about 2.2 times chance each.
  - Half the training data -0.033; improvised-only training -0.055 on improvised test speech.
  - 160 runs repeated from night 1 were identical.
- Analyses, selection records and test reports are in C:\Affective Computing\affect\artifacts\ (analysis, selections, test_reports, analyses\night2_20260926, and rebuilt sweep records in sweeps).
- Code in the canonical tree: tree digest 9a535d6d... (132 tests pass in the Cowork container, on the PC, and on the node with nothing skipped).
- Literature: 62 papers read in full on 2026-09-26 (claude/IEMOCAP literature review.md).
- Open decision: final test numbers from the runs as they are (train 3 sessions, pick the epoch on the 4th, test the 5th; what Pepino et al. 2021 did) or from retraining on 4 sessions for a fixed number of epochs (what the plan first said; needs new training code). Night 2's speaker-out split, which trains on 4 sessions, scored 0.564 against session out's 0.559, so the 4th session adds little for this CNN.
- Proposed next (none started): cnn_confirm_2 on new seeds 6 to 10; a tuned row for every split; speed perturbation as a setting; the literature splits; gender as an auxiliary task; the CNN + Transformer model; slides for the meeting.
- Deferred: TPOT on openSMILE features (section 5); an optional GPU-yield check for workers (nothing that runs in the background is added without Harrison asking).
