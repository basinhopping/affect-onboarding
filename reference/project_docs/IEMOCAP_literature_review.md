# IEMOCAP literature review: splits, preprocessing, models and results

Written 2026-09-26, during the night-2 run. It covers the 62 PDFs Harrison pulled into C:\Affective Computing\papers\ (60 of the 67 reading-list entries, plus 2 stand-in papers). It answers Dr. Luitel's request to research how IEMOCAP papers split and preprocess the data, and what models and results they report. It also verifies the literature claims in our other project pages (section 14).

Marks: [checked] means I re-checked the number or wording in the paper's own text. [reader] means a reader found it while reading the paper page by page and confirmed it on the page image, and I did not re-check it myself. Nothing here comes from an abstract alone. The per-paper notes, one file per paper with page numbers, are in C:\Affective Computing\papers\notes\.

## 1. Early findings

1. **Our CNN sits where comparable published models sit.** CNN-type models trained from scratch, speaker independent, 4 classes, score 56.6 to 61.8 UA (5.1). Our night-1 baseline has a validation UA of 0.589 over 5 seeds with 3 training sessions, where most of those papers train on 4. The best cnn_confirm_1 candidates reach about 0.61 on single seeds so far. Higher published numbers come from pretrained models (70 to 78 UA fine-tuned), from recurrent or attention layers with heavy augmentation (64 to 68 UA), or from leakier protocols.
2. **Most published IEMOCAP numbers are not clean.** Of the 35 papers that run their own IEMOCAP experiments, 8 let test data choose epochs or settings for at least one reported number, 14 describe no validation set or do not say how it was used, and 13 describe a validation set kept apart from the test data (3.2). 9 of the 35 report spread over repeated runs. Our protocol (a validation session, test read once after selection, 5 seeds with intervals) is stricter than nearly all of them.
3. **Leakage has measured sizes.** On the same model, random splits add about 6.5 WA points over speaker-independent splits (Wang 2021, Gat 2022). Using the test fold as the validation set adds 2.7 (session out) to 4.7 WA points (speaker out) (emotion2vec).
4. **Her gender splits would be new results.** None of the 62 papers trains on one gender and tests on the other. Routing speakers to a model of their own gender helped by 3.6 to 4.3 UA on German corpora (Vogt and André 2006).
5. **Shared scripts inflate text models a lot and audio models little** (Pepino 2020, Gamage 2018).
6. **Speed perturbation** (training copies at 0.9x and 1.1x speed) added 2.3 and 3.5 UA to IEMOCAP CNNs (Aldeneh 2017, Latif 2019). Our pipeline does not have it yet.
7. **Tuning:** random search plus small full grids is what the tuning literature supports. TPOT cannot tune a neural network, and Hyperband gains little on jobs as short as ours.
8. **Normalization layers:** at our batch sizes batch normalization is the right default. Group normalization only helps at 8 or fewer examples per batch, and SubSpectral normalization is a cheap audio-specific option.

## 2. How I found this

- Harrison pulled 62 PDFs (999 pages). I extracted each paper's text with page markers and matched every file to the reading list by title, authors and arXiv number. 60 of the 67 entries are covered. Entry #12 is covered by Gamage's PhD thesis, whose Chapter 4 is the ICASSP 2017 paper. Two files turned out to be different papers from the ones listed, and I read them anyway (section 16).
- Eleven readers (Claude subagents) each took a group of papers by theme. Each read every page and wrote one notes file per paper with the same sections: data, split and model selection, preprocessing, model and training, results, gender, claims to check, and a one-line table row. Every number taken from a table was confirmed on the page image, because text extraction can scramble table columns.
- I then re-checked against the paper's own text every number that appears in the findings and tables below (the [checked] mark). Where the text was ambiguous I looked at the page image, for example Pepino 2020's Table 1, which prints decimal commas.
- I sorted the 35 papers with their own IEMOCAP experiments into validation categories myself, from each paper's own description (3.2). A first tally from the readers' labels was too harsh: it said only 4 papers keep test data out, and the correct count is 13.

## 3. How IEMOCAP is split

### 3.1 Protocol families

| Family | What is held out | How validation is chosen | Papers (main protocol) |
|---|---|---|---|
| Session out, 5 folds | one session (2 speakers) | a training session (Pepino 2021, ours); inner cross-validation (Sahu 2017); 20% of the training data (emotion2vec and EmoBox, which do not say how it is drawn) or a random 20% of training utterances (Morais 2022); none or not stated (SUPERB, SpeechFormer, Li 2019, Zou 2022, Chen and Rudnicky 2023, PTS-SNN, Tarantino 2019, Pepino 2020) | 13 |
| Session out, split by speaker | one speaker of the held-out session | the other speaker of that session | Satt 2017 |
| Speaker out, 10 folds | one speaker | the test speaker's session partner (Aldeneh 2017, Latif 2019); one held-out speaker (Wang 2021); a random 20% of training (Gamage 2018, neural chapters); none or not stated (Mariooryad 2014, Cai 2021, Latif 2022, Latif 2023, Antoniou 2023) | 9 |
| Other speaker-independent | 2 test speakers per fold (Gat 2022); 2 fixed test speakers (Han 2014); 1 fixed test speaker (Ibrahim 2024) | from the non-test speakers | 3 |
| Session 5 as test | Session 5 | 10% of training dialogues (DialogueGCN); not stated (DialogueRNN); cross-validation inside Sessions 1 to 4, then retraining (EMOVOME); Session 4 (Zhang 2019) | 4 |
| Random or speaker mixed | random utterances or folds | none (Zhu 2022 GLAM, Tang 2024, Light-SERNet, Yue 2024); one fold, use not stated (Sutherland 2021) | 5 |
| Cross-corpus only | all of IEMOCAP as a test set | the training corpus's validation set | Triantafyllopoulos 2025, Wagner 2023 (not in the 35) |

Several papers also report a second protocol: emotion2vec and Zou 2022 report both session out and speaker out; Wang 2021, Gat 2022 and Latif 2022 report a random or speaker-mixed split beside a speaker-independent one; Chen and Rudnicky 2023 also use Session 5 as test with shrinking training data.

Antoniou 2023 recommends "a 10-fold Speaker Independent cross-validation with one speaker as test, eight as training and one as validation set" [checked], and calls the 5-fold session-out design equally reasonable, with 10 folds scoring slightly higher [checked]. They do not say how the validation speaker is chosen [reader].

### 3.2 Validation and test selection (my sorting of the 35 papers)

- **Test data chose epochs or settings for at least one reported number (8):** Pepino 2020 (epochs and architecture tuned on the cross-validated results, which the authors call possibly optimistic); Gamage 2018 (Chapter 3 settings picked on the test folds); Tarantino 2019 (window chosen on test-session loss); Zhu 2022 GLAM (no validation set, mixup strength picked on test results); Li 2019 (best of 100 epochs kept, with no validation set); Cai 2021 (auxiliary-loss weight chosen on the test folds); Su 2026 PTS-SNN (headline equals the peak of sweeps run on IEMOCAP); Chen and Rudnicky 2023 (runs filtered on performance, with no validation set). emotion2vec also measures this effect on purpose (3.3).
- **No validation set described, or its use not described (14):** Antoniou 2023 (its reproductions), SUPERB, Mariooryad 2014, Sutherland 2021, DialogueRNN, Yue 2024, Light-SERNet, SpeechFormer, Zou 2022, Latif 2023, Tang 2024, Satt 2017, Ibrahim 2024, Latif 2022.
- **Validation kept apart from the test data (13):** Pepino 2021 (a training session); Wang 2021 and Gat 2022 (held-out speakers); emotion2vec and EmoBox (20% of the training data, drawn in an unstated way) and Morais 2022 (a random 20% of training utterances); DialogueGCN (10% of training dialogues); EMOVOME (cross-validation inside Sessions 1 to 4); Aldeneh 2017 and Latif 2019 (the test speaker's partner); Han 2014 (development speakers); Sahu 2017 (inner cross-validation); Zhang 2019 (Session 4).

Partner validation (Aldeneh 2017, Latif 2019, Satt 2017) keeps test labels out of selection, but the validation speaker sits in the same conversations as the test speaker, so validation data is unusually close to test data. Our next-session validation avoids that.

Spread over repeated runs on IEMOCAP is reported by Pepino 2021 (5 seeds, SD), Pepino 2020 (10 seeds, interquartile range), Aldeneh 2017 (SD over test speakers), Latif 2019, Latif 2022 and Latif 2023 (10 repetitions), Zhu 2022 (100 random splits), Yue 2024 (20 runs) and Sutherland 2021 (SD over folds): 9 of 35.

### 3.3 What leakage and selection are worth

| Paper | Model | Comparison | Numbers | Mark |
|---|---|---|---|---|
| Wang 2021 | HuBERT Large, partly fine-tuned | random split vs 10-fold speaker out | 79.58 vs 73.01 WA (+6.6) | [checked] |
| Gat 2022 | HuBERT Large | random split vs 5 folds of 2 test speakers | 78.3 vs 71.9 WA (+6.4) | [checked] |
| Latif 2022 | semi-supervised adversarial autoencoder | 10 folds with speakers mixed vs speaker out | 68.8 vs 66.7 UA (+2.1) | [checked] |
| emotion2vec | emotion2vec | test fold also used as validation, session out | 74.48 vs 71.79 WA (+2.7) | [checked] |
| emotion2vec | emotion2vec | test fold also used as validation, speaker out | 77.64 vs 72.94 WA (+4.7) | [checked] |
| Zou 2022 | wav2vec 2.0 with co-attention | 10-fold speaker out vs 5-fold session out | 72.70 vs 71.05 UA (+1.7) | [checked] |
| Pepino 2020 | audio model | random folds vs session folds, AvAUC read from a bar chart | about 0.83 to 0.84 vs 0.785 | [reader] |
| Pepino 2020 | BERT text model | session folds vs session and script folds, AvAUC from the same chart | about 0.87 vs 0.79 | [reader] |

The Zou row is not pure leakage: speaker-out folds train on more speakers and keep the test speaker's partner, and so their conversations, in training.

### 3.4 Class sets and counts

- The standard set is 4 classes with happy and excited merged, 5,531 utterances. Antoniou 2023 recommends it [checked]. Busso 2008 supports the merge: its raters confused happy and excited (0.18 and 0.16 in its confusion table), and the authors merged them for their own agreement analysis [reader].
- Antoniou 2023 prints the counts as {1708, 1636, 1103, 1084} for {neutral, happy, sad, angry} [checked]. Our label files give angry 1,103 and sad 1,084, so the paper has those two swapped.
- Other sets in use: happy without excited, 4,490 utterances (Sahu 2017, Tarantino 2019); 5 classes (Han 2014, whose fifth class is named inconsistently); 6 classes in conversation work (DialogueGCN, DialogueRNN: 5,810 training and validation utterances and 1,623 test [checked]); 10 classes on one session (Yue 2024); 3-level arousal, valence and dominance (Zhang 2019).
- Busso 2008: "ten thousand and thirty nine turns" (5,255 scripted) and approximately twelve hours [checked]. Three of six raters per turn; a categorical label only when the top vote is unique, which 74.6% of turns reach [reader].

## 4. Preprocessing conventions

| Paper | Input | Frame (window / hop) | Normalization | Length | Augmentation |
|---|---|---|---|---|---|
| Aldeneh 2017 | 40 log-mel filterbanks (openSMILE) | 25 ms | per speaker, including the test speaker's own statistics | whole utterance, padded to multiples of 32 frames | speed 0.9 and 1.1 |
| Satt 2017 | linear spectrogram, 0 to 4 kHz, 10 Hz bins | 40 ms / 10 ms | not stated | pieces of at most 3 s; the test averages the pieces | none |
| Latif 2019 | raw waveform | none | not stated | 6 s best of 1 to 8 s; edge silence removed | speed 0.9 and 1.1 |
| Latif 2023 | 128 log-mel | 40 ms / 10 ms | not stated | 7.5 s, cut or padded | speed, mixup, SpecAugment |
| Li 2019 | mel spectrogram with deltas | 50 ms / 25 ms | not stated | 7.5 s, cut or padded | none |
| SpeechFormer 2022 | 128 log-mel, spectrogram, or wav2vec features | 25 ms / 10 ms | not stated | fixed, about 6.5 s | none stated |
| Zou 2022 | 40 MFCC, spectrogram, wav2vec 2.0 | 40 ms / 10 ms | not stated | 3 s segments | none |
| Zhu 2022 GLAM | MFCC | not stated | not stated | 2 s segments, 1.6 s overlap, averaged | mixup |
| Tang 2024 | MFCC from 26 mel filters | not stated | not stated | 1.8 s segments, averaged | mixup |
| Light-SERNet 2022 | 40 MFCC | 64 ms windows | waveform scaled to [-1, 1] | 3 s or 7 s | none |
| Tarantino 2019 | IS09 or eGeMAPS frame features | 25 ms / 10 ms | statistics of neutral data | 500-frame windows | none |
| Triantafyllopoulos 2025 (CNN14) | 64 log-mel | 32 ms / 10 ms | not stated | random 3 s crop each epoch; full utterance at test | none |
| Pepino 2021 | wav2vec 2.0 Base, all layers weighted | 20 ms frames | per speaker (test speaker's data) or global | whole utterance | none |

All preprocessing entries are [reader] unless marked elsewhere. Takeaways:

- 16 kHz is universal. Log-mel with a 25 ms / 10 ms frame is the most common spectral front end, with 40, 64 and 128 bins all in use. Our screen draws 32, 40, 48 and 64 mel bins; 80 and 128 are worth adding.
- Lengths are fixed crops of 3 to 7.5 s or whole utterances. Several papers cut long utterances into segments and average the segment posteriors at test time. Latif 2019 found 6 s best for raw audio. Our current screen draws 4, 5 and 7.5 s training crops and tests on whole utterances.
- Speaker normalization with the test speaker's own statistics is common (Aldeneh 2017, Pepino 2021: 67.2 vs 65.8 UA global [checked]). It assumes unlabeled audio from the test speaker is available at test time, which fits the "user checks a box" deployment, but it must be labeled whenever it is reported.
- Speed perturbation is the only augmentation with measured gains on IEMOCAP CNNs: Aldeneh 2017 59.5 to 61.8 UA, Latif 2019 56.72 to 60.23 UA [checked].

## 5. Models and results

### 5.1 Trained from scratch, speaker independent

| Paper | Model | Split | UA | WA | Mark |
|---|---|---|---|---|---|
| Satt 2017 | CNN on spectrogram, improvised only | session out, partner validates | 56.6 | 66.1 | [checked] |
| Satt 2017 | CNN + LSTM, improvised only | same | 59.4 | 68.8 | [checked] |
| Latif 2019 | CNN on raw audio | speaker out, partner validates | 56.72; 60.23 with speed perturbation | not reported | [checked] |
| Aldeneh 2017 | CNN on 40 log-mel | speaker out, partner validates | 59.5; 61.8 with speed perturbation | not reported | [checked] |
| SpeechFormer 2022 | hierarchical Transformer on log-mel | session out, no validation | 59.8 | 58.2 | [checked] |
| Latif 2022 | semi-supervised autoencoder, gender and speaker auxiliary tasks | speaker out | 66.7 (64.5 single task) | not stated | [checked] |
| Latif 2023 | CNN + BLSTM + attention, augmentation, multitask | speaker out, 10 repetitions | 68.1 +- 1.5 (64.3 baseline) | not stated | [checked] |

Classical features:

| Paper | Features and classifier | Split | UA | Mark |
|---|---|---|---|---|
| Gamage 2018 | IS09 (384 features), linear SVM | speaker out | 56.75 | [checked] |
| Mariooryad 2014 | IS11 descriptors, SVM, lexical normalization | speaker out, no validation | 56.73 (55.73 without) | [checked] |
| Sahu 2017 | IS10 (1,582 features), SVM, happy not merged | session out, inner cross-validation | 57.88 | [checked] |
| Latif 2019 | GeMAPS and eGeMAPS, SVM | speaker out | 57.92 and 58.76 | [checked] |
| Gamage 2018 | MFCC i-vectors, GPLDA | speaker out | 58.0 | [checked] |

### 5.2 Pretrained models

| Paper | Model | Mode | Split | Result | Mark |
|---|---|---|---|---|---|
| SUPERB 2021 | HuBERT Large, HuBERT Base, wav2vec 2.0 Large, wav2vec 2.0 Base | frozen, weighted layers | 5 folds of the "standard splits" | accuracy 67.62, 64.92, 65.64, 63.43 | [checked] |
| WavLM 2022 | WavLM Large, Base+, Base | frozen, SUPERB setup | SUPERB | accuracy 70.62, 68.65, 65.94 | [checked] |
| EmoBox 2024 | Whisper large v3, WavLM Large, HuBERT Large, wav2vec 2.0 Base | frozen last layer | 5 folds of 2 speakers, 20% of training validates | UA 73.54, 69.47, 67.42, 58.27 | [checked] |
| Pepino 2021 | wav2vec 2.0 Base | frozen, all layers weighted | session out, a training session validates, 5 seeds | UA 67.2 +- 0.7 (speaker norm), 65.8 (global) | [checked] |
| emotion2vec 2024 | emotion2vec (pretrained on IEMOCAP audio among others) | frozen, linear head | session out, 20% of training validates | WA 71.79, UA 72.69 | [checked] |
| EMOVOME 2024 | UniSpeech-SAT Large | frozen, linear head | Session 5 test | UA 69.58 | [checked] |
| Chen and Rudnicky 2023 | wav2vec 2.0 | fine-tuned: plain, TAPT, P-TAPT | session out, 4 sessions train, no validation, 15 epochs | UA 69.9, 73.5, 74.3 | [checked] |
| Wang 2021 | HuBERT Large, partly fine-tuned | fine-tuned | 10-fold speaker out | WA 73.01 | [checked] |
| Morais 2022 | HuBERT and wav2vec 2.0 fused; wav2vec 2.0 alone | fine-tuned | session out, random 20% validates | UA 77.76; 77.07 | [checked] |
| Cai 2021 | wav2vec 2.0 Base with a speech-recognition auxiliary task | fine-tuned | speaker out, auxiliary weight chosen on test | WA 78.15 | [checked] |
| Su 2026 PTS-SNN | spiking head on emotion2vec features | frozen backbone | session out, no validation described | UA 73.72 | [checked] |

### 5.3 Leaky or not comparable (reference only)

| Paper | Model | Protocol problem | Number | Mark |
|---|---|---|---|---|
| Zhu 2022 GLAM | multi-scale CNN | random 80/20 repeated 100 times, no validation | UA 73.90 (72.56 with an 8:1:1 split) | [checked] |
| Tang 2024 | CNN + Transformer | random 80/20, no validation | UA 72.72 | [checked] |
| Light-SERNet 2022 | lightweight CNN | 10 folds of unstated construction, fixed 300 epochs | UA 70.76 | [checked] |
| Li 2019 | BLSTM with attention pooling and a gender auxiliary task | best of 100 epochs, no validation | UA 82.8 | [checked] |
| Tarantino 2019 | self-attention on IS09 | window chosen on test loss, happy not merged | UA 63.8 | [checked] |
| Han 2014 | DNN + extreme learning machine | 5 classes | UA 48.2 | [checked] |
| Yue 2024 | differential evolution, gender-aware | one session, 10 classes, random 10 folds | WA 56.3 | [reader] |

Cross-corpus, trained on MSP-Podcast and tested on all of IEMOCAP: Triantafyllopoulos 2025 reports UA .617 for a tuned wav2vec 2.0 Large, .544 for HuBERT Large, .445 to .519 for CRNNs, and .491 to .503 for static eGeMAPS or IS11 functionals with an MLP [checked]. Wagner 2023 reports CCC on IEMOCAP of .663 arousal, .518 dominance and .448 valence [reader].

## 6. Gender

- None of the 62 papers trains on one gender and tests on the other.
- Vogt and André 2006 (German acted speech from the Berlin database and spontaneous SmartKom speech, Naive Bayes): routing each utterance to a classifier of the speaker's true gender raised UA by 4.3 points (Berlin) and 3.6 (SmartKom) [reader]. With automatic gender detection, about 90% accurate, the combined system improved the overall rate by 2 to 4% [checked]. Caveat: features were selected on test-set accuracy [reader].
- Lee and Narayanan 2005 built separate male and female models on call-center speech but never compared them with a mixed model, so the claim that it shows gender-specific models winning, as Vogt cites it, is not in that paper [reader].
- Gender as an auxiliary training task: Latif 2022, speaker out, 66.7 vs 64.5 UA [checked]; Li 2019, 82.8 vs 72.6 UA, but with test-picked epochs [checked].
- Fairness: Gorrostieta 2019 (MSP-Podcast only) found female speakers scored lower and narrowed the gap with a training penalty [reader]. Wagner 2023 found valence favors female speakers [reader]. EMOVOME reports male-minus-female UA differences on IEMOCAP Session 5, and notes that two test speakers allow no conclusion [reader].
- Our own first data point (validation, night-1 baseline, one mixed model): female 0.594 vs male 0.575, not consistent across sessions (protocols page, section 8).

So the "user checks a gender box" routing has one supportive precedent, on other corpora, and cross-gender transfer on IEMOCAP is untested in this literature.

## 7. Scripts and lexical overlap

- Busso 2008: the actors "were asked to perform three selected scripts" [checked], and each session was recorded twice with the same pair [reader]. Every session shares the script text.
- Pepino 2020 compared random, session, and session-plus-script folds (script 3 tested, the other scripts in training). The BERT text model dropped from about 0.87 AvAUC (session folds) to about 0.79 (script-disjoint), while the audio model stayed at about 0.785 in both; random folds raised audio to about 0.83 to 0.84 [reader, read from a bar chart]. The authors recommend avoiding repeated scripted dialogues between training and test, or discarding scripted dialogues, especially for text systems [checked].
- Gamage 2018 (speaker out): word-based features scored 64.0 UA on all test data and 52.2 on improvised-only test utterances; fused with IS09 acoustics, 68.3 and 66.6; phoneme features lost about 2.4 [checked].
- Sutherland 2021 (text only): accuracy 67.5 on all of IEMOCAP and 62.7 on improvised only [checked].
- For our audio CNN, little inflation from scripts is expected. Improvised and scripted results are already reported separately in test-report. A script-disjoint split matters for the text and pretrained stages.

## 8. Tuning methods: grid, random, TPE, Hyperband, TPOT

- Bergstra and Bengio 2012: "Random searches of 8 trials match or outperform grid searches of (on average) 100 trials" [checked]. Only 1 to 4 of 7 hyperparameters mattered per data set [reader]. They report searches with random-experiment efficiency curves, the best of n trials against n [reader], and suggest running each cell of a small grid on the same random sample of the other settings [reader]; our cnn_grid_1 holds the other settings at the base values instead.
- TPE (Bergstra et al. 2011): a model-based search that beat random and manual search on its tasks, for example 14.13% error against 18.97% and 18.63% on the convex task [checked], with caveats: one run per method, and TPE searched wider bounds [reader].
- Optuna (Akiba et al. 2019): define-by-run search spaces, TPE sampling, and asynchronous successive-halving pruning, which raised completed trials per 4 hours from 35.8 to 1,278.6 in their test [checked].
- Hyperband (Li et al. 2018): over an order of magnitude faster than its competitors on CIFAR-10 [checked], but "did not provide significant speedup" where fixed per-configuration overhead dominates, as on 117 small OpenML data sets [checked]. Our jobs take 1 to 3 minutes including data loading and evaluation, so early stopping of weak configurations would gain less.
- TPOT (Olson et al. 2016): genetic programming over scikit-learn pipelines, population 100 for 100 generations [checked]; no neural network operators [reader]. Each pipeline is scored on the held-out 25% of one 75/25 split, and that score appears to be the one reported, so results may be optimistic [reader]. TPOT-NN (Romano et al. 2021) adds only a PyTorch logistic regression and a multilayer perceptron [checked]; convolutional search is left as future work, and it beat plain TPOT significantly on 2 of 6 data sets at up to 18 times the runtime [reader].

For Dr. Luitel: our design (random search over many factors, full grids on the few that matter, then 5-seed confirmation) is what this literature recommends. TPOT is useful as a classical AutoML baseline on fixed acoustic features such as eGeMAPS functionals, where published SVM baselines sit at 56.8 to 58.8 UA, not as a tuner for the CNN.

## 9. Normalization layers and statistics

- Group normalization (Wu and He 2018, ImageNet): error for batch norm vs group norm at 32, 16, 8, 4 and 2 images per GPU was 23.6 vs 24.1, 23.7 vs 24.2, 24.8 vs 24.0, 27.3 vs 24.2, and 34.7 vs 24.1 [checked]. At our batch sizes (16 to 32) batch norm is about 0.5 better; group norm wins from 8 down. The default is 32 groups [reader]. This matters later for the pretrained stage, where batches are small.
- SubSpectral normalization (Chang et al. 2021): batch norm computed separately per frequency sub-band. It gained 1.3 and 0.9 points on acoustic scenes with 2 sub-bands and 1.3 and 1.0 on keyword spotting with 4; the gain disappears at 8 to 16 sub-bands [reader]. A cheap setting to add to the CNN.
- None of the 62 papers compares normalization layers for emotion recognition, so any result we get there is new.
- Bouthillier et al. 2021: P(A > B) is the share of paired runs in which A beats B. With a percentile-bootstrap interval, a difference is significant when the lower end is above 0.5 and meaningful when the upper end is above 0.75, and about 29 runs per model are needed for gamma 0.75 with alpha and beta 0.05 [checked]. Resampling the data is the largest source of variance, and varying only the seed is the worst estimator [reader]. Our comparisons pair seeds and folds and resample validation utterances in the bootstrap, which covers part of this; resampling the training data in backed comparisons would cover more.

## 10. Spiking networks: settings to start from

- Bittar and Garner 2022 (speech commands; no IEMOCAP): filterbank features were "directly fed into various networks" as real values [checked]. Leak trainable per neuron within 0.60 to 0.96, threshold 1, reset by subtraction, a boxcar surrogate gradient (0.5 within 0.5 of the threshold), 2 hidden layers plus a readout, Adam at 1e-3 [reader]. The readout sums the softmax of the output potentials over time; in their own SHD comparison spike count was slightly better for one model, 87.45 vs 87.27 [checked]. Best spiking 94.51% against best non-spiking 95.06% on Speech Commands v2 [checked].
- Eshraghian et al. 2023 (tutorial): U[t] = beta U[t-1] + W X[t] - S[t-1] theta; reset by subtraction preferred; the arctan surrogate is the snnTorch default and the threshold defaults to 1; memory for backpropagation through time grows with time steps times neurons; normalizing inputs helps, and learnable thresholds do not [reader].
- Su 2026 PTS-SNN is the only spiking paper here that uses IEMOCAP: 73.72 UA on frozen emotion2vec features, session out, no validation described [checked]; emotion2vec's pretraining data includes IEMOCAP [checked]; a plain spiking network on the same features reached only 41.78 WA [checked].
- Mansouri-Benssassi and Ye 2021 use STDP on RAVDESS and eNTERFACE, with no IEMOCAP [reader].

## 11. Pretrained models: facts for step 5

- wav2vec 2.0: Base has 12 layers of width 768 (95M parameters), Large 24 of 1024 (317M); pretrained on LibriSpeech 960 h or LibriVox 53.2k h [reader].
- HuBERT: Base on LibriSpeech 960 h; Large and X-Large on Libri-Light 60k h; 95M, 317M and 964M parameters [reader].
- WavLM Large: 94k hours (Libri-Light 60k, GigaSpeech 10k, English VoxPopuli 24k) [checked]; frozen SUPERB emotion accuracy 70.62 [checked].
- emotion2vec pretrains on five corpora including IEMOCAP, 262 hours in total [checked], so it stays out of headline results (plan 4.5).
- EmoBox uses frozen last-layer features only and lists that as a limitation [reader]; Pepino 2021 and SUPERB weight all layers.
- Triantafyllopoulos 2025: learning rate 1e-4 with batch 4 made several self-supervised models collapse to chance; lower rates are worth trying [reader].

## 12. Splits to consider for the next run

Tonight's run already covers: session out; speaker out with partner validation; speaker out with the partner in training; conversation-grouped; pooled random; within session; cross-speaker within a session; female-only and male-only training; half training data; improvised-only and scripted-only training; and the four other class sets. Dr. Luitel's split ideas were suggestions to explore, so the goal is one baseline-CNN row per option, not a decision.

| Candidate | Design | What it measures | Precedent |
|---|---|---|---|
| A. Session out, 4 sessions, random validation | the 4 training sessions minus a stratified random 20% for validation; one test session | what a 4th training session adds; the protocol most pretrained-model papers use | emotion2vec, EmoBox, Morais 2022 |
| B. Test session as validation | 4 sessions train; early stopping on the test session itself | how much test-set selection inflates a CNN result; a labeled demonstration, never a result | emotion2vec's starred rows |
| C. Session 5 as test | Sessions 1 to 4 minus 10% of their dialogues for validation; Session 5 test; 4 and 6 classes | comparability with the conversation literature | DialogueGCN, DialogueRNN, EMOVOME, Zhang 2019 |
| D. Training-size curve | session out with 25%, 50% (tonight) and 75% of the training utterances | how data-limited the CNN is | Chen and Rudnicky 2023 |
| E. Script-disjoint test | test one script in the held-out session, with that script removed from training | lexical leakage; matters most for text and pretrained models | Pepino 2020 |
| F. Within-speaker folds | train, validate and test inside one speaker's utterances (10 speakers, 5 folds each) | the speaker-dependent ceiling; the "enroll the user" case | my proposal; not in these papers |
| G. Cross-corpus | train on IEMOCAP, test on a second corpus over the shared classes | generalization | Triantafyllopoulos 2025, Wagner 2023, Latif 2022 and 2023; waits for a second dataset |

B needs one change in the split code, which today refuses any fold whose validation and test sets overlap. That refusal would stay for every other split.

## 13. Other additions suggested by the reading

- Speed perturbation as an augmentation setting (0.9x and 1.1x copies), per Aldeneh 2017 and Latif 2019.
- A test policy that cuts long utterances into fixed segments and averages their posteriors (Satt 2017, Zhu 2022, Tang 2024, Latif 2022), beside our whole-utterance test.
- n_mels of 80 and 128 in the screen (128 is used by SpeechFormer, Latif 2023 and, in effect, Li 2019).
- Gender as an auxiliary training task (Latif 2022, Li 2019): a third way to use gender, beside separate models and routing.
- SubSpectral normalization as a normalization-layer option.
- A TPOT baseline on eGeMAPS functionals, in its own environment.
- Per-speaker UA and its spread inside each fold (Triantafyllopoulos 2025 report this), which our per-utterance predictions already allow.

## 14. Status of the claims in our other pages

Protocols page (claude/IEMOCAP protocols and split plan.md):

1. Busso 2008, 10,039 turns and about twelve hours: confirmed [checked]. Three scripts in every session: confirmed by clear implication, not in one sentence [checked for "three selected scripts"].
2. Speaker overlap inflates results by 5 to 7 points: confirmed, Wang +6.6 WA and Gat +6.4 WA [checked]. Gat's two numbers are plain HuBERT Large; its speaker normalization is adversarial training against a speaker-identification head, not feature normalization [reader].
3. Selecting on the test fold adds 2.7 to 4.7 points: confirmed; the two pairs are +2.69 (session out) and +4.70 (speaker out) [checked].
4. Script overlap helps text but not audio (Pepino 2020): confirmed; the script-disjoint folds test only script 3 [reader].
5. Session 5 as a fixed test set in conversation work, 6 classes: confirmed, 5,810 training and validation utterances and 1,623 test [checked]. Session 5 is implied by training on "the first eight speakers", not named [reader].
6. No IEMOCAP paper trains on one gender and tests on the other: confirmed within these 62 papers.
7. Vogt and André 2006, gender-dependent beats gender-independent: confirmed with caveats (German corpora, features chosen on test accuracy).
8. Reference ranges. "CNNs from scratch UA 57.6 to 61.8": corrected to 56.6 to 61.8 (Satt 56.6, Aldeneh 61.8); 57.6 was not found. "eGeMAPS or ComParE with SVM 58.7 to 61.7": corrected to 56.8 to 58.8 (Gamage IS09 56.75, Latif 2019 eGeMAPS 58.76); the 61.7 in Latif 2019 is a cited CNN result. "Fine-tuned self-supervised models 69.9 to 77.8": the endpoints are confirmed (Chen and Rudnicky 69.9, plain fine-tuning with no validation set; Morais 77.76, a two-model fusion) [checked].
9. Pepino 2021, 67.2 +- 0.7: wav2vec 2.0 Base, pretrained only, all layers weighted, 5 seeds, UA; the speaker normalization uses the test speaker's data [checked].
10. TPOT cannot search neural architectures: confirmed; TPOT-NN adds only logistic regression and an MLP [checked].
11. No head-to-head of normalization layers for emotion recognition: confirmed within these 62 papers.
12. EmoFilm: the fifth class is contempt; the paper excluded surprise and disgust [checked]. It prints English 360 and Spanish 342, but its own per-emotion averages give 343 and 359, our file counts [reader]. Its mean length of 3.5 s (SD 1.2) conflicts with our median of 1.87 s [reader].

Plan page (claude/Affective Computing Plan and Data Map.md, version 2.7):

- Antoniou 2023 lists angry and sad swapped: confirmed [checked].
- Wang 2021, 79.58 vs 73.01 WA: confirmed [checked].
- emotion2vec, 71.79 vs 74.48 WA: confirmed [checked].
- EmoBox UA and WA for Whisper large v3, WavLM Large, HuBERT Large and wav2vec 2.0 Base: all confirmed [checked]. EmoBox's folds are "5 folds of 2 distinct speakers", with 20% of the training data for validation, drawn in an unstated way [reader].
- WavLM's 94k hours: confirmed [checked].
- Bittar and Garner: "directly fed into various networks" confirmed [checked]. The readout sentence is confirmed, but their own SHD table shows spike count slightly ahead for one model [checked].
- Bouthillier et al.: the 0.5 and 0.75 thresholds and the 29 runs are confirmed [checked].
- Neumann and Vu 2017, 56.10 WA: the paper is missing; Tarantino 2019 quotes it as WA 56.1 [checked, secondhand].
- Padi et al. 2021, Cawley and Talbot 2010, the Spikformer energy figures and the SpikingJelly benchmark are not in this set and stay unverified here.

## 15. Other checks

- Pepino 2020 notes that seed changes can reorder systems, which supports our multi-seed rule [checked].
- Triantafyllopoulos 2025 found that model rankings change with hyperparameters and that newer or bigger models are not reliably better [reader]. Comparing CNN, CNN + Transformer and CNN + LIF therefore needs the same search budget for each and ranges over configurations and seeds, not single numbers.

## 16. Papers missing or substituted

- Not available: #42 Du et al. 2025 (spiking network with CNN), #61 Xia and Liu 2015, #63 Akçay and Oğuz 2020, #66 Atmaja and Sasou 2023.
- Wrong file: #22 Neumann and Vu 2017 (the file is Mountzouris et al. 2023, Electronics, with no IEMOCAP; the right paper is arXiv 1706.00612); #60 Xia and Liu 2017 (the file is Latif et al. 2022, IEEE Transactions on Affective Computing, read anyway).
- Not pulled: #39 Neftci, Mostafa and Zenke 2019 (arXiv 1901.09948).
- Substitute: #12 is Gamage's 2018 PhD thesis; its Chapter 4 is the ICASSP 2017 paper, whose printed title is "Salience based Lexical Features for Speech Emotion Recognition" [reader].

## 17. Files read

| # | Paper | File | Pages |
|---|---|---|---|
| 1 | Busso et al. 2008, IEMOCAP: Interactive emotional dyadic motion capture database | Busso_2008_iemocap.pdf | 30 |
| 2 | Antoniou et al. 2023, Designing and Evaluating Speech Emotion Recognition Systems: A Reality Check Case Study with IEMOCAP | 2304.00860v1.pdf | 5 |
| 3 | Pepino et al. 2021, Emotion Recognition from Speech Using wav2vec 2.0 Embeddings | pepino21_interspeech.pdf | 5 |
| 4 | Pepino et al. 2020, Fusion Approaches for Emotion Recognition from Speech Using Acoustic and Text-Based Features | 2403.18635v1.pdf | 5 |
| 5 | Wang et al. 2021, A Fine-tuned Wav2vec 2.0/HuBERT Benchmark for Speech Emotion Recognition, Speaker Verification and Spoken Language Understanding | 2111.02735v3.pdf | 7 |
| 6 | Gat et al. 2022, Speaker Normalization for Self-Supervised Speech Emotion Recognition | 2202.01252v2.pdf | 5 |
| 7 | Ma et al. 2024, emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation | 2312.15185v1.pdf | 12 |
| 8 | Triantafyllopoulos et al. 2025, Charting 15 years of progress in deep learning for speech emotion recognition: A replication study | 2508.02448v1.pdf | 18 |
| 9 | Ma et al. 2024, EmoBox: Multilingual Multi-corpus Speech Emotion Recognition Toolkit and Benchmark | 2406.07162v1.pdf | 5 |
| 10 | Yang et al. 2021, SUPERB: Speech processing Universal PERformance Benchmark | 2105.01051v4.pdf | 6 |
| 11 | Mariooryad and Busso 2014, Compensating for speaker or lexical variabilities in speech for emotion recognition | Mariooryad_2014.pdf | 11 |
| 12 | Gamage 2018, Speech-Based Emotion Recognition: Linguistic and Saliency-Based Systems (PhD thesis, UNSW), standing in for the ICASSP 2017 paper | public version.pdf | 266 |
| 13 | Sutherland et al. 2021, arXiv 2103.03700 (script overlap) | 2103.03700v1.pdf | 2 |
| 14 | Ghosal et al. 2019, DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation | D19-1015.pdf | 11 |
| 15 | Majumder et al. 2019, DialogueRNN: An Attentive RNN for Emotion Detection in Conversations | 1811.00405v4.pdf | 8 |
| 16 | Vogt and André 2006, Improving Automatic Emotion Recognition from Speech via Gender Differentiation | 392_pdf.pdf | 4 |
| 17 | Lee and Narayanan 2005, Toward detecting emotions in spoken dialogs | leesap2005.pdf | 11 |
| 18 | Gorrostieta et al. 2019, Gender De-Biasing in Speech Emotion Recognition | gorrostieta19_interspeech.pdf | 5 |
| 19 | Yue et al. 2024, Advanced differential evolution for gender-aware English speech emotion recognition | s41598-024-68864-z.pdf | 11 |
| 20 | Gómez-Zaragozá et al. 2024, EMOVOME: A Dataset for Emotion Recognition in Spontaneous Real-Life Speech | 2403.02167v3.pdf | 22 |
| 21 | Aldeneh and Mower Provost 2017, Using regional saliency for speech emotion recognition | 2017_Aldeneh_ICASSP.pdf | 5 |
| 23 | Satt et al. 2017, Efficient Emotion Recognition from Speech Using Deep Learning on Spectrograms | satt17_interspeech.pdf | 5 |
| 24 | Latif et al. 2019, Direct Modelling of Speech Emotion from Raw Speech | 1904.03833v4.pdf | 5 |
| 25 | Aftab et al. 2022, Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition | 2110.03435v1.pdf | 5 |
| 26 | Han et al. 2014, Speech Emotion Recognition Using Deep Neural Network and Extreme Learning Machine | han14_interspeech.pdf | 5 |
| 27 | Sahu et al. 2017, Adversarial Auto-Encoders for Speech Based Emotion Recognition | 1806.02146v1.pdf | 5 |
| 28 | Tarantino et al. 2019, Self-attention for Speech Emotion Recognition | tarantino19_interspeech.pdf | 5 |
| 29 | Zhu et al. 2022, Speech Emotion Recognition with Global-Aware Fusion on Multi-scale Feature Representation | 2204.05571v1.pdf | 7 |
| 30 | Eyben et al. 2016, The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing | The_Geneva_Minimalistic_Acoustic_Parameter_Set_GeMAPS_for_Voice_Research_and_Affective_Computing.pdf | 13 |
| 31 | Wagner et al. 2023, Dawn of the Transformer Era in Speech Emotion Recognition: Closing the Valence Gap | 2203.07378v4.pdf | 20 |
| 32 | Chen et al. 2022, SpeechFormer: A Hierarchical Efficient Framework Incorporating the Characteristics of Speech | 2203.03812v2.pdf | 5 |
| 33 | Li et al. 2019, Improved End-to-End Speech Emotion Recognition Using Self Attention Mechanism and Multitask Learning | li19n_interspeech.pdf | 5 |
| 34 | Zou et al. 2022, Speech Emotion Recognition with Co-Attention Based Multi-Level Acoustic Information | 2203.15326v1.pdf | 5 |
| 35 | Cai et al. 2021, Speech Emotion Recognition with Multi-Task Learning | cai21b_interspeech.pdf | 5 |
| 36 | Latif et al. 2023, Multitask Learning from Augmented Auxiliary Data for Improving Speech Emotion Recognition | 2207.05298v1.pdf | 12 |
| 37 | Triantafyllopoulos et al. 2022, Probing speech emotion recognition transformers for linguistic knowledge | 2204.00400v2.pdf | 5 |
| 38 | Tang et al. 2024, Speech Emotion Recognition via CNN-Transformer and multidimensional attention mechanism | 2403.04743v2.pdf | 14 |
| 40 | Eshraghian et al. 2023, Training Spiking Neural Networks Using Lessons From Deep Learning | 2109.12894v6.pdf | 61 |
| 41 | Su et al. 2026, PTS-SNN | 2602.08240v1.pdf | 10 |
| 43 | Mansouri-Benssassi and Ye 2021, Generalisation and robustness investigation for facial and speech emotion recognition using bio-inspired spiking neural networks | s00500-020-05501-7.pdf | 14 |
| 44 | Bittar and Garner 2022, A surrogate gradient spiking baseline for speech command recognition | fnins-16-865897.pdf | 18 |
| 45 | Baevski et al. 2020, wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations | 2006.11477v3.pdf | 19 |
| 46 | Hsu et al. 2021, HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units | 2106.07447v1.pdf | 10 |
| 47 | Chen et al. 2022, WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing | 2110.13900v5.pdf | 14 |
| 48 | Morais et al. 2022, Speech Emotion Recognition using Self-Supervised Features | 2202.03896v1.pdf | 5 |
| 49 | Chen and Rudnicky 2023, Exploring Wav2vec 2.0 Fine Tuning for Improved Speech Emotion Recognition | 2110.06309v3.pdf | 5 |
| 50 | Park et al. 2019, SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition | 1904.08779v3.pdf | 6 |
| 51 | Bergstra and Bengio 2012, Random Search for Hyper-Parameter Optimization | bergstra12a.pdf | 25 |
| 52 | Bergstra et al. 2011, Algorithms for Hyper-Parameter Optimization | NIPS-2011-algorithms-for-hyper-parameter-optimization-Paper.pdf | 9 |
| 53 | Akiba et al. 2019, Optuna: A Next-generation Hyperparameter Optimization Framework | 1907.10902v1.pdf | 10 |
| 54 | Li et al. 2018, Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization | 16-558.pdf | 52 |
| 55 | Olson et al. 2016, Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science | 1603.06212v1.pdf | 8 |
| 56 | Romano et al. 2021, TPOT-NN: augmenting tree-based automated machine learning with neural network estimators | nihms-2035489.pdf | 25 |
| 57 | Wu and He 2018, Group Normalization | Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf | 17 |
| 58 | Chang et al. 2021, SubSpectral Normalization for Neural Audio Data Processing | 2103.13620v1.pdf | 5 |
| 59 | Bouthillier et al. 2021, Accounting for Variance in Machine Learning Benchmarks | MLSys-2021-accounting-for-variance-in-machine-learning-benchmarks-Paper.pdf | 23 |
| 62 | Zhang et al. 2019, Attention-augmented End-to-end Multi-task Learning for Emotion Prediction from Speech | 1903.12424v1.pdf | 5 |
| 64 | Abbaschian et al. 2021, Deep Learning Techniques for Speech Emotion Recognition, from Databases to Models | sensors-21-01249-v2.pdf | 27 |
| 65 | Parada-Cabaleiro et al. 2018, Categorical vs Dimensional Perception of Italian Emotional Speech | paradacabaleiro18_interspeech.pdf | 5 |
| 67 | Ibrahim et al. 2024, What Does it Take to Generalize SER Model Across Datasets? | 2406.09933v1.pdf | 5 |
| extra | Latif et al., Multi-Task Semi-Supervised Adversarial Autoencoding for Speech Emotion Recognition, IEEE TAC 2022 (stand-in file for #60) | 76270876.pdf | 15 |
| extra | Mountzouris, Perikos, Hatzilygeroudis, Speech Emotion Recognition Using Convolutional Neural Networks with Attention Mechanism, Electronics 2023 (stand-in file for #22) | electronics-12-04376.pdf | 31 |

Per-paper notes: C:\Affective Computing\papers\notes\<number>_<name>.md, with the reading instructions and this index in the same folder.
