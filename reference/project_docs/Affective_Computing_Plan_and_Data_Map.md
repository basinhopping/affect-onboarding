# Affective Computing: Plan and Data Map

> Copy for the onboarding kit, 2026-10-12: Harrison's node login, the node's internal address, the contents of his home folder on the node and the names of his earlier project's folders are left out. Everything else is as written on 2026-09-25 (version 2.7); later decisions are in the other project docs.

Version 2.7, 2026-09-25. Replaces versions 2 to 2.6 of the same day and version 1 (2026-09-24). Status: building has started. Build steps 1 and 2 are done (13): the repository `C:\Affective Computing\affect` exists with its tests passing, and the training copy of IEMOCAP is assembled and verified (8.1). Step 3, the environment on the node, is next. `HANDOFF.md` in the repository has the exact commands.

## What changed from version 1

- Phase 0 ran on your computer. Every data fact in section 1 is read from the audit files in `C:\Affective Computing\_phase0_audit\` (the folder was `C:\Effective Computing` until the rename on 2026-09-25; its contents are unchanged, VERIFICATION_LOG entry 2). That folder's `README.md` is the verification log: it is generated from the JSON outputs, so no number in it was typed by hand. Version 1's layout and counts came from the public release, and all of them held.
- Found: 501 empty audio files in your IEMOCAP copy. The collaborator's `wav` folder fills all of them, verified two independent ways.
- Found: his Session 5 spectrograms show the wrong utterances (1,228 of 1,241). His Sessions 1 to 4 are right.
- The node has 4 GPUs, one tmux session each.
- New: how every result gets backed (section 3), both LIF variants in detail (4.4), the modular code design (section 6).
- 2.1: decided to use his verified copies of the 501 files (1.3) and to regenerate his image format from audio instead of training on his files (1.4). Notebooks run headless on the node from shell scripts, as in your last project (7.1).
- 2.2: read the summer project's latest node copy (`Downloads\small`). Its failures are now rules with the evidence beside each (6.7). Also new: the exact MPS fix and thread caps (8.3), stage gates and priority in the queue (5.1), and the old notebook command (7.1).
- 2.3: the live node check (8.3): 4 H100 NVL, the fix still needed and working, 40 GB free on a network drive. New: scp moves one checked archive each way (8.2), a fresh environment with current library facts (8.3), and a storage budget (8.4).
- 2.4: the second node check: AlmaLinux 10.2, 256 CPU threads, 1,511 GiB of RAM, a local disk with 316 GB free, and `/data` and `/scratch` present. The storage budget now leans on RAM (8.4).
- 2.5: the third node check: your home folder has a limit of about 95 GB (40 GB free); `/scratch` is a separate local disk with 337 GB free that you can write to; `/tmp` and `/var/tmp` are emptied on a timer. The node layout is now concrete (8.4).
- 2.6: no cleanup rule for `/scratch` was found on the node, and you're fine keeping rebuildable things there, so the node layout is final (8.4). The remaining small choices are written down as defaults (12).
- 2.7: the project is now Affective Computing, the repository and package are `affect`, and nothing uses `/scratch`: the environment, the model cache and the data live in your home folder (8.3, 8.4). Answered: the full IEMOCAP release is 24.0 GB, which your node home could hold only by leaving about 16 GB for everything else, and 19.9 GB of it is motion capture and video, which nothing planned reads, so only the 1.44 GB training copy goes up (8.4). Built: the repository with its tests (6.2, 6.6), and the data: all three received folders copied into `data\iemocap\raw\` with a hash for every file, and the training copy assembled and verified three times (8.1). New findings: video exists only for Session 1, motion capture is incomplete, and a fifth stray `.pk` file (1.2, 1.5). Changed from 2.6: `local_paths.json` is gone, because the data sits inside the repository on both machines; three node scripts were added (`common.sh`, `setup_env.sh`, `check.sh`).

---

## 1. The data, verified

### 1.1 IEMOCAP copy

| Item | Value |
|---|---|
| Labeled utterances | 10,039, every ID unique |
| Dialogs and speakers | 151 dialogs, 10 speakers; every dialog has both an F and an M speaker |
| Audio format | every readable sentence wav is mono, 16-bit, 16 kHz, uncompressed |
| Hours from label times | 12.44 for all utterances, 6.99 for the four classes |
| Evaluators | exactly 3 categorical evaluators on every utterance |

All labels:

| xxx | fru | neu | ang | sad | exc | hap | sur | fea | oth | dis |
|---|---|---|---|---|---|---|---|---|---|---|
| 2,507 | 1,849 | 1,708 | 1,103 | 1,084 | 1,041 | 595 | 107 | 40 | 3 | 2 |

Four classes, excited merged into happy:

| Session | neu | hap+exc | ang | sad | Total |
|---|---|---|---|---|---|
| 1 | 384 | 278 | 229 | 194 | 1,085 |
| 2 | 362 | 327 | 137 | 197 | 1,023 |
| 3 | 320 | 286 | 240 | 305 | 1,151 |
| 4 | 258 | 303 | 327 | 143 | 1,031 |
| 5 | 384 | 442 | 170 | 245 | 1,241 |
| All | 1,708 | 1,636 | 1,103 | 1,084 | 5,531 |

The class balance differs a lot between sessions (Session 4 has 327 angry and 143 sad), so per-fold results will differ for that reason alone. Angry 1,103 and sad 1,084 also settles a disagreement between papers: Antoniou et al. 2023 list those two counts swapped, and Padi et al. 2021 have them right.

By kind: improvised 2,943 (neu 1,099, hap+exc 947, sad 608, ang 289) and scripted 2,588 (ang 814, hap+exc 689, neu 609, sad 476). So 814 of the 1,103 angry utterances are scripted. A model could partly learn "scripted" as a stand-in for angry, so every result gets broken down by kind.

Durations of the four-class utterances (readable files): median 3.575 s, mean 4.512 s, 95th percentile 10.809 s, maximum 29.13 s. 14.1% are longer than 7.5 s and 6.3% longer than 10 s. This sets the crop lengths in the sweep.

### 1.2 Layout and label format

```
IEMOCAP_full_release/
  README.txt, Documentation/
  Session1/ ... Session5/
    dialog/
      EmoEvaluation/        LABELS. One summary .txt per dialog
        Attribute/, Categorical/, Self-evaluation/   per-evaluator files
      transcriptions/       one .txt per dialog
      wav/                  whole-dialog recordings, stereo, 16 kHz
      lab/                  turn timings
      avi/, MOCAP_*/        video and motion capture (not needed)
    sentences/
      wav/<dialog>/         AUDIO. One .wav per utterance
      ForcedAlignment/, MOCAP_*/   (not needed now)
```

The release's own `README.txt` says the labels are in `dialog/Evaluation/`. On disk the folder is `EmoEvaluation/`.

A real excerpt from `Session1/dialog/EmoEvaluation/Ses01F_impro01.txt`:

```
% [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]

[6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
C-E2:	Neutral;	()
C-E3:	Neutral;	()
C-E4:	Neutral;	()
C-F1:	Neutral;	(curious)
A-E3:	val 3; act 2; dom  2;	()
A-E4:	val 2; act 3; dom  3;	(mildly aggravated but staying polite, attitude)
A-F1:	val 3; act 2; dom  1;	()
```

`C-E` lines are the three external evaluators, `C-F1` or `C-M1` is the actor rating themselves, and `A-` lines are valence, activation and dominance ratings. The released label is exactly the unique plurality over the external evaluators' listed categories, for 10,039 of 10,039 utterances. That means soft labels can be built from the same votes and stay consistent with the hard labels.

Naming, from the release's `README.txt`: "Ses01F_impro01_M000 indicates first session, Female actor is wearing markers, actors are performing improvisation 1 and this is the first utterance of the Male actor." So the speaker is the letter in the last field, confirmed on disk.

Scenarios: `impro01` to `impro07` appear in all five sessions, `impro08` in sessions 2 to 5, split variants `impro05a/b` and `impro08a/b` only in Session 3, `script01` to `script03` (in parts) in all five sessions, and `script01_1b` only in Session 5.

Two traps:

- The per-evaluator files in `EmoEvaluation/Attribute/` store valence and activation reversed (6 minus the value) relative to the summary files, for all 21,437 ratings compared. The summary files follow the published scale: mean activation is 3.64 for angry and 2.56 for sad. Anything we build reads the summary files only.
- Four stray `.pk` files sit inside `sentences/wav`, and a fifth in `Session2/dialog/wav` (found in 2.7). Nothing reads them.

### 1.3 The 501 empty files, and the collaborator's `wav` folder

501 sentence wavs in your copy are 0 bytes, all in eight Session 5 dialogs (`Ses05M_script01_1` from turn M020 onward, then all of `Ses05M_script01_1b` to `Ses05M_script03_2`). 249 of them are four-class utterances: 20% of Session 5's four-class data, and all from scripted dialogs. Training or testing on this copy as it is would silently drop them. Only `sentences/wav` is affected: the dialog recordings, label files, transcripts and lab files for all five sessions contain no empty files.

The collaborator's `wav` folder is not one dialog. It is the complete Session 5 sentence audio: all 31 dialogs, 2,170 files. 1,669 are byte-identical to your copy and 501 are the files your copy is missing. He most likely sent it because of this gap.

The 501 were checked two independent ways:

1. Their lengths match the label times to within 0.14 ms.
2. Each one is a sample-exact cut of its dialog recording at the labeled start time. The same test was first run on known-good files (80 of 80 exact), then on every utterance in the dataset: 10,039 of 10,039 are exact cuts of their dialog recordings.

So the audio-to-label mapping is verified for the whole dataset, and the fix is safe.

Decided 2026-09-25: the assembled dataset (8.1) takes the 501 files from his folder, and the manifest records that source for each of them.

### 1.4 His spectrograms

| Check | Result |
|---|---|
| Files | 5,531 PNG, exactly the four-class IDs, none missing, none extra |
| Format | 226 x 226 pixels, grayscale, no axes or margins, saved by Matplotlib |
| Time axis | stretched or squeezed to 226 columns, so a 0.6 s and a 29 s utterance have the same width |
| Settings (inferred) | best match of four candidates: linear-frequency STFT, n_fft 2048, hop 512, dB clipped 80 below the maximum. These are close to librosa's defaults. His code was not seen |
| Sessions 1 to 4 | every image (4,290 of 4,290) matches its own utterance's audio best |
| Session 5 | 1,228 of 1,241 images show a different utterance. In 645 of those the audio shown has a different emotion label, and in 416 it is not a four-class utterance at all |
| Duplicates | 12 pairs of byte-identical images under different IDs, all in Session 5 |

How the Session 5 result was measured: our own rendering of each utterance's audio was correlated with every one of his images from the same session. In Sessions 1 to 4 each image matches its own audio with correlation at least 0.94. In Session 5 the images match other utterances just as strongly (median 0.966). The shift between an image's name and the audio it shows grows by about one position per dialog through the first 21 dialogs (from 2 to 21), then drops to one position for the last 10. That pattern suggests his file list had an extra entry per dialog folder, but the cause is an inference.

What this means: training on his files as they are would give Session 5 wrong labels, and fold 5 of the evaluation would be meaningless. Decided 2026-09-25: we don't train on his files. Our code regenerates images in his format from the audio for all five sessions, as one arm of the sweep. That tests whether squeezing a spectrogram into a fixed-size image costs accuracy, which is a finding either way. The audit README is the evidence to share with him.

Questions for him, now narrower:

1. Can you send the script that made the spectrograms?
2. Which file list did it use for Session 5? The images there are shifted against their names.
3. Does "cleaned" mean anything beyond keeping the four classes? The set of IDs is exactly the four-class set.
4. Did you know about the 501 empty Session 5 files, and is that why you sent the `wav` folder?

### 1.5 Not covered by the audit

- `avi/`, `MOCAP_*`, `sentences/ForcedAlignment` and `Documentation/corpus.dic` were not scanned for empty files. We don't need them now. Listing the whole release in 2.7 found that video (`dialog/avi`, 29 files) exists only for Session 1, and that motion capture covers 150 of 151 dialogs and 9,949 to 10,006 of the 10,039 utterances, depending on the stream (VERIFICATION_LOG entry 3).
- His spectrogram code was not seen. The settings in 1.4 are inferred.
- Why Session 5 went wrong is inferred from the offset pattern only.

---

## 2. Evaluation protocol

Fixed before any training and applied to every number. You agreed to these in version 1. The only change is that counts now come from our files.

| Choice | Rule |
|---|---|
| Split | leave one session out, 5 folds; never a random split |
| Validation | inside the training sessions only: test session k uses session k+1 for validation (5 uses 1), training on the other three |
| Final runs | the chosen configuration is retrained on all four training sessions for the epoch count validation picked, then tested once. The three-session early-stopped model is reported beside it as a check |
| Class set | four classes, happy and excited merged, 5,531 utterances |
| Subsets | primary: improvised and scripted together; secondary: improvised only (2,943). Never compared with each other |
| Metrics | primary UA (mean per-class recall); also WA, macro F1, per-class recall, confusion matrix, and every metric broken down by kind |
| Aggregation | each metric per fold; headline is the mean of the five folds, each weighted equally; pooled metrics are stored and always labeled "pooled" |
| Secondary protocol | ten folds leaving out one speaker, reported as a separate table |

Why these:

- **Random splits inflate results.** They put the same actors and conversations in training and test. Wang et al. 2021 ran one partially fine-tuned HuBERT Large both ways: 79.58% WA on random splits, 73.01% speaker-independent.
- **The test fold must never pick the model.** emotion2vec's own table: 71.79% WA with a separate validation set, 74.48% when the test fold also served as validation. It matters more here than in most projects, because picking the best of hundreds of sweep configurations by test score would report the luckiest one.
- **The class set alone moves results.** Padi et al. 2021, one model: 60.63, 64.65 and 62.82 UA for three class sets. Every number carries its class set. The merged set is what EmoBox and emotion2vec use, so ours compare directly.
- **Why not ten speaker folds as primary.** There the validation speaker is the test speaker's dialog partner, so model selection sees the same conversations as the test. Holding out whole sessions keeps them apart.
- **Scripts and scenarios repeat across sessions** (1.2), so the same words recur in every fold. That matters little for audio-only models and a lot once transcripts are added.

Baselines and controls, all run with the CNN in the pilot:

| Control | What it tells us | Expected |
|---|---|---|
| Chance | UA floor | 25% UA |
| Always predict neutral | WA floor | 30.9% WA overall (1,708 of 5,531); per fold from the table in 1.1 |
| Label shuffle | the pipeline leaks if this beats chance | about 25% UA |
| Random split against leave one session out | size of the speaker effect | random split higher |
| Duration, loudness and pitch statistics with logistic regression | how much simple cues already give | unknown |

---

## 3. How every result gets backed

Your rule: every result is backed, with at least 5 seeds and its spread or confidence interval. Exploratory runs use one seed. When candidates are close, they get a seeded comparison. Here is that rule made precise.

### 3.1 Two kinds of runs

| | Screening run | Backed run |
|---|---|---|
| Purpose | decide what to test next | produce a number for a meeting or the paper |
| Seeds | 1 | at least 5, never including the screening seed |
| Reads | validation only | validation, then test once |
| Can appear in the paper | only as a description of the search | yes |

Anything reported as a finding gets backed, including factor effects. If "128 mel bins beat 64" goes on a slide, both settings are run with 5 seeds first.

### 3.2 What a backed number is

- One seed means one full five-fold run. Its value is the fold mean of UA.
- Over 5 seeds we report the mean, the standard deviation, and a 95% confidence interval using the t distribution with 4 degrees of freedom (critical value 2.776, so the half-width is 1.24 times the SD).
- Next to it: the per-fold means across seeds (session-to-session variation), and the minimum and maximum seed.

### 3.3 Comparing two models

Both models run on the same folds, and every per-utterance prediction is saved, so the comparison can be paired.

1. **Paired bootstrap for the difference.** Resample seeds, and within each fold resample test utterances (the same resample for both models), then recompute the difference in fold-mean UA. 10,000 draws give a 95% interval for the difference. This covers both training randomness (seeds) and the finite test set (utterances).
2. **Probability of outperforming**, following Bouthillier et al. 2021: P(A > B) across matched runs, with a percentile bootstrap interval. They call a result significant when the interval's lower end is above 0.5 and meaningful when its upper end is above 0.75.
3. **Families of comparisons** get a Holm correction. For example, the best configuration of each of the five model variants compared pairwise is 10 comparisons.

Proposed rule: A beats B when the 95% interval of the difference excludes zero after correction. Everything else is reported as "no detectable difference", with the numbers.

One sentence for you to decide on: Bouthillier et al. recommend about 29 trainings per model for the P(A > B) test to reach a 5% false-negative rate, so for the few head-to-head claims the paper rests on, we could go past 5 seeds.

### 3.4 When screening results get promoted

The pilot runs the baseline CNN with 5 seeds, which measures seed noise directly. After a screening round, every configuration within 2 seed standard deviations of the best, and at least the top 5 of each model variant, gets 5 seeds. Proposed: if the measured seed SD of fold-mean UA is above 1.0 point, screening itself moves to 3 seeds, because a single seed could not rank configurations that far apart.

### 3.5 Selection bias

Picking the best of many configurations on validation overestimates it. Cawley and Talbot (2010) found this effect "often of comparable magnitude to differences in performance between learning algorithms". Guards:

- Selection reads validation only.
- The chosen configurations are written to a selection file before any test file is read. The analysis code refuses to read test results for a sweep that has no selection file, so the order is recorded, not remembered.
- Final seeds are new seeds.

### 3.6 Findings registry

One file, `findings.csv`: finding ID, the claim in one sentence, the runs and tables behind it, the test used, the result, and a status (supported, not supported, or pending). A sentence in the paper cites a finding ID. If a number is not in a findings row, it does not go in the paper.

---

## 4. Models, one at a time

Order: CNN, then CNN + Transformer, then CNN + LIF (both variants), then the pretrained model. Each variant's sweeps run overnight while the next one is being built.

### 4.1 Shared input stage

16 kHz mono audio (verified), log-mel computed on the GPU or read from a cache built once per feature setting. Settings exposed: window and hop in ms, n_fft, n_mels, f_min, f_max, log offset, normalization mode, training crop length, test-time policy (whole utterance or averaged windows), augmentation.

### 4.2 CNN

Convolution blocks (convolution, batch norm, ReLU, pooling) over the mel-by-time image, pooling over time and frequency, then a linear classifier. Settings: number of blocks, channels per block, kernel size, pooling size, dropout, final pooling (mean, max, attention or statistics), classifier width. For reference, published log-mel CNNs under this protocol: 56.10% WA (Neumann and Vu 2017) and 62.82% UA (Padi et al. 2021, pretrained on speaker data).

### 4.3 CNN + Transformer

The CNN shrinks the spectrogram, each remaining time step becomes a vector of size d_model, a Transformer encoder models the sequence, then pooling and a classifier. Settings: the CNN front end plus strides, d_model, number of layers, number of heads, feed-forward size, dropout, attention dropout, activation, pre-norm or post-norm, positional encoding, pooling.

### 4.4 CNN + LIF: both variants

**What an SNN is, in plain terms.** A normal network passes real numbers between layers. A spiking network passes spikes, 0 or 1, over a series of time steps. A leaky integrate-and-fire (LIF) neuron keeps a running value, its membrane potential. Each time step the potential shrinks by a factor beta (the leak), adds its input, and when it crosses a threshold the neuron outputs a 1 and resets. A spike is a step function, which has no useful gradient, so training uses a surrogate gradient: on the backward pass a smooth curve stands in for the step. Audio suits this, because spectrogram frames already form a time axis.

**How audio enters.** In the speech SNN work, the usual approach feeds each spectrogram frame as one time step, injecting the real-valued features straight in as input current. Bittar and Garner 2022 write that filterbank features were "directly fed into various networks". The alternative repeats the whole spectrogram for T steps, which is the vision-style approach. Both are tested (encoding is a sweep factor).

| | Variant A: hybrid | Variant B: fully spiking |
|---|---|---|
| Front end | standard CNN blocks, as in 4.2 | spiking convolutions: LIF after every convolution |
| Time axis | CNN output frames become the SNN time steps (optionally pooled in time first) | each spectrogram frame is one time step, or the static spectrogram repeated T steps |
| Spiking part | 1 to 3 LIF layers, feedforward or recurrent | the whole network |
| Readout | non-spiking output layer; logits from the membrane potential summed over time, or spike counts (sweep factor) | same |

Bittar and Garner compared readouts on speech commands and found the cumulative sum of the output potential best; it "gave the best performance and is what is used in all presented results". On that task their best spiking network reached 94.51% against 95.06% for the best non-spiking one. Expect the SNNs to trail the CNN somewhat. Their usual selling point is efficiency.

Settings for both: beta (fixed or learned per layer), threshold, reset (subtract or zero), surrogate function and slope, time pooling factor, encoding, time steps for the static encoding, hidden sizes, recurrence, dropout, batch norm, readout, activity regularization.

Efficiency, reported as secondary metrics: spike rate per layer and synaptic operations (firing rate times time steps times the layer's operations), converted to energy with the 45 nm figures most SNN papers use (4.6 pJ per multiply-accumulate, 0.9 pJ per accumulate, from Horowitz 2014, as used in Spikformer). These estimates ignore memory access, which critics point out, so they are labeled as estimates.

Known difficulties, with the planned fix for each:

| Problem | Fix |
|---|---|
| Training memory grows with time steps: about 500 steps for 5 s at a 10 ms hop | time pooling or a longer hop as sweep factors; fused multi-step kernels; activation checkpointing if needed |
| Silent layers that never spike | checked on every run (spike rate is logged); initialization and threshold as sweep factors |
| Library defaults hiding settings | every LIF argument is passed explicitly, and a unit test checks our layer against a plain reference implementation on fixed inputs |

Library: SpikingJelly's fused multi-step kernels are much faster than stepping in Python. In its own benchmark at 32 steps, forward plus backward took 41.87 ms in plain PyTorch against 3.37 ms with its Triton backend. snnTorch is simpler and widely taught. The choice waits for the node's Python and PyTorch versions. Either way the layer is verified against the reference implementation.

Presentation: both variants are built, swept and backed the same way, and the professor sees both with numbers. We don't pick one on her behalf.

### 4.5 Pretrained model

All candidates read raw 16 kHz audio. The EmoBox numbers come from a small head trained on frozen backbone features, leave one session out, merged four classes.

| Model | Pretraining data | EmoBox IEMOCAP UA / WA | Risk that it saw IEMOCAP |
|---|---|---|---|
| Whisper large v3 (encoder) | large web-collected set, contents not listed | 73.54 / 72.86 | unknown; IEMOCAP is licensed, so unlikely |
| WavLM Large | 94k hours: Libri-Light, GigaSpeech, VoxPopuli (WavLM paper) | 69.47 / 69.07 | none from the stated data |
| HuBERT Large | Libri-Light (not opened) | 67.42 / 66.69 | none from the stated data |
| wav2vec 2.0 Base | LibriSpeech (not opened) | 58.27 / 57.73 | none from the stated data |
| emotion2vec | 262 hours from five corpora, IEMOCAP included (emotion2vec paper) | not in EmoBox's table; 71.79 WA in its own paper | high: pretrained on IEMOCAP audio |
| emotion2vec+ | fine-tuned on EmoBox data (seed) or undisclosed pseudo-labeled data (base, large) | no IEMOCAP number on the card | high or unknown |

Recommendation: WavLM Large as the main model (clean pretraining data, strong published numbers). The Whisper large v3 encoder is the comparison arm: it is the highest of all ten models in EmoBox's IEMOCAP table, and Whisper is also the likely transcript source for the chatbot later. HuBERT Large is a third. emotion2vec stays out of headline results because of its IEMOCAP pretraining.

Any checkpoint on the Hugging Face Hub that is already fine-tuned for emotion on IEMOCAP is excluded, because it was trained on sessions we test on. The SpeechBrain IEMOCAP card, for example, says it "is trained on IEMOCAP training data" and does not say which split.

Modes to sweep: frozen backbone with a small head (last layer as in EmoBox, or a learned weighted sum of layers), partial fine-tuning of the top layers, and full fine-tuning. Wang et al. 2021 report 73.01% WA speaker-independent for partially fine-tuned HuBERT Large, under a ten-fold protocol not directly comparable to EmoBox.

---

## 5. Sweeps on the 4 GPUs

### 5.1 Queue and workers

One job is one JSON spec: the complete settings, the fold and the seed. Sweeps are written into a queue folder. On the node there are 4 tmux sessions, one per GPU, each running `worker.sh <gpu>`. A worker takes the next spec with an atomic file move, so no job is taken twice. It runs the job, writes into that run's own folder, and moves the spec to `done` or `failed`. Small jobs can run several to a GPU (a slots setting per worker). A worker can check free GPU memory before taking a job, since the node is shared, and can stop taking new jobs at a set morning hour. You start the four sessions whenever you like; overnight is the default.

Two more pieces, both kept from what worked in `lane.sh` and `finish.sh`:

- **Gates.** A gate is a queue item that waits until every job of a named sweep has finished or failed, then runs a registered step, for example the selection for the next stage, and enqueues that stage. A stage can never start on half-finished results. Unlike the old barriers, a worker waiting on a gate keeps taking any other job that is ready, so no GPU sits idle.
- **Priority.** Jobs carry a priority, so the most important work runs first and a night that ends early has done what mattered most.

`python -m affect.cli status` is the morning check: jobs done, failed and waiting per sweep, the failed jobs' last error lines, and any stop condition that fired. It reads the queue folders and run folders, not the logs, so a library warning containing the word "failed" can't be miscounted as a failure.

### 5.2 Stages for each model variant

| Stage | Contents | Seeds |
|---|---|---|
| Pilot (CNN only, first night) | environment check on all 4 GPUs; data manifest check; one job per fold timed; baseline CNN with 5 seeds (measures seed noise); controls from section 2 | 5 for the baseline, 1 for the rest |
| Screening | one factor at a time around a baseline (one figure per factor), plus random search over all factors (interactions); selection on validation UA | 1 (or 3, per 3.4) |
| Seeded comparison | promoted configurations (3.4) | 5, new seeds |
| Final | the chosen configuration, retrained as in section 2, tested once | 5 or more, new seeds |

If a stage does not fit in one night it runs over two. Seeds and folds are never cut to make it fit.

### 5.3 Screening factors

Data and features, screened on the CNN first:

| Factor | Values | Why |
|---|---|---|
| Input | log-mel; log linear spectrogram; MFCC with deltas; image format matching his (regenerated from audio) | representation; the last arm tests fixed-size images |
| n_mels | 40, 64, 80, 128 | frequency detail |
| Window / hop | 25 / 10 ms; 50 / 12.5 ms | time against frequency resolution |
| f_max | 8 kHz; 4 kHz | whether the upper band helps |
| Normalization | per utterance; global from training sessions; per speaker; none | loudness is an arousal cue, and per-utterance normalization removes it |
| Training crop | 3, 5, 7.5, 10 s; whole utterance | lengths vary (1.1); 14.1% run past 7.5 s |
| Augmentation | none; SpecAugment at two strengths; added noise; mixup; speed perturbation | speed perturbation also shifts pitch, itself an emotion cue, so it is tested rather than assumed |
| Class balance | none; weighted loss; balanced sampler | unequal classes, and unequal per session |
| Labels | hard majority; soft labels from the three evaluators' votes | three evaluators often disagree |

Model settings per variant are listed in section 4. Optimization: learning rate (log grid), weight decay, batch size, schedule, early-stopping patience, label smoothing. AdamW and bf16 throughout. Per-speaker normalization at test time uses the test speaker's unlabeled audio; if it is used, that is stated with the result.

### 5.4 Cost

Four GPUs over a ten-hour night give 40 GPU-hours. Per-job times are guesses until the pilot measures them: a few minutes per CNN fold, tens of minutes per fold for full WavLM Large fine-tuning. The pilot's timings set how many random-search draws a night holds.

### 5.5 Stop conditions for the pilot

You accepted these in version 1. The first two are new.

1. The data manifest does not show 10,039 non-empty wavs and 5,531 four-class utterances matching the Phase 0 index: stop.
2. Any of the 4 GPUs fails the environment check (CUDA not visible after the daemon fix, or a training step fails): stop.
3. CNN validation UA below 50%: a pipeline problem. Stop and debug.
4. Label-shuffle UA above 30%: something leaks. Stop.
5. Any audio-only UA above 80% under leave one session out: investigate before believing it.

---

## 6. Code architecture

### 6.1 Principles

- All logic lives in a Python package. Notebooks hold settings, calls and displays only.
- A new model or dataset is a new module plus one registry entry. Nothing else changes.
- No setting has a default anywhere in the package. A missing setting is an error, and so is an unknown one.
- One job writes one folder, and nothing overwrites anything. No code chooses its own output path: the path comes from the sweep and the settings hash.
- A variant is a setting, never a copy of the tree or of a script.
- No program edits code, notebooks or settings files. The repository root holds no scripts, and a test fails if a .py file appears outside `src/` or `tests/`.
- Numbers shown anywhere are read from files that record how they were made.

### 6.2 Layout

As built on 2026-09-25 (steps 1 and 2 of 13). Folders marked "step 4" exist and are empty until the CNN.

```
affect/                       one canonical tree: C:\Affective Computing\affect
  README.md  HANDOFF.md  VERIFICATION_LOG.md  findings.csv  requirements.in
  notebooks/                  thin: settings, calls, displays (step 4)
  src/affect/
    __init__.py               the MPS fix, applied on import; refuses if torch came first uncleared
    env.py                    environment and GPU checks, require_gpus
    settings.py               schema validation (both directions), hashing
    registry.py               @register_dataset, @register_feature, @register_model
    catalog.py                one import line per component module
    hashing.py, paths.py      file and settings hashes; every location, fixed inside the repository
    data/                     base.py, iemocap.py, splits.py, audio.py, manifest.py,
                              ingest.py, assemble.py, transfer.py, walk.py
    eval/                     metrics.py, stats.py (CIs, paired bootstrap, P(A>B), Holm)
    runs/                     store.py (output folders), provenance.py; spec.py with step 4
    features/ augment/ models/ train/ sweep/ analysis/   step 4 onward
    cli.py                    python -m affect.cli ingest | assemble | verify-data | data-summary | env-check
                              (run | worker | enqueue | collect | status from step 4)
  tests/                      python -m unittest discover -s tests -t .
  scripts/                    node: common.sh, setup_env.sh, check.sh, unpack.sh, pack.sh,
                              and from step 4 worker.sh, enqueue.sh, report.sh;
                              your PC: push.ps1, pull.ps1   (nothing else)
  data/                       not code, not in git; see data/README.md
    iemocap/raw/              verified copies of the received folders, never edited
    iemocap/manifests/        path, bytes and SHA-256 of every raw file
    iemocap/assembled/        what training reads; the only data that goes to the node
  logs/                       one log per worker and per notebook run (node only)
  queue/                      pending/ running/ done/ failed/ (node only, step 4)
  artifacts/                  written on the node, pulled to the PC
    checks/                   verify-data and env-check reports
    env/                      the environment's exact package versions
    sweeps/<sweep_id>/sweep.json
    runs/<sweep_id>/<run_id>/
    selections/  tables/  figures/
    notebooks/                dated, executed copies from report.sh
    analyses/<name>/<id>/     one-off checks and audits, each run in its own folder
```

The data sits inside the repository on both machines (`data/iemocap/assembled/`), so no machine needs a data path and 2.6's `local_paths.json` was dropped. Node-only settings (environment, model cache, MPS fix, thread caps) live in one file, `scripts/common.sh`.

### 6.3 Settings

Each component declares its settings as a dataclass with no defaults: the feature extractor, the model, the training loop, the augmentations. A settings dict from a notebook is checked against every declaration. A missing field raises, an unknown field raises, and a wrong type raises. The validated settings are hashed, and the hash names the run. This makes "every setting visible" enforced by the code: a setting the notebook does not show cannot reach a model.

### 6.4 Adding a model or a dataset

- **Model:** one file in `models/` with its settings dataclass and a build function, registered by name. It declares whether it takes a spectrogram or a waveform, and the data stage supplies the right one. A notebook is copied from the template.
- **Dataset:** one file in `data/` that builds a standard utterance table (ID, dataset, speaker, fold group, raw label, mapped label, duration, audio path, transcript if any), declares its label mapping into the shared label set, and declares its fold scheme. IEMOCAP's module holds everything in section 1: the 501 fills, the summary-file rule, the speaker rule.
- **Cross-corpus:** a test dataset setting. The model trains on IEMOCAP (with its validation session) and is tested on all of the other corpus, over the labels both share. Nothing else changes.

### 6.5 Run folders

Each job writes into `<run_id>.tmp/` and renames it to `<run_id>/` only when complete. A failed job becomes `<run_id>.failed-<time>/`. A complete folder is never touched again, and re-queuing a finished job skips it.

| File | Content |
|---|---|
| `spec.json` | every setting, the fold, the seed, the sweep ID |
| `provenance.json` | git commit, entry point, start and end time, host, GPU, Python, torch and CUDA versions, data manifest hash |
| `metrics_val.json` | per-epoch curves and final validation metrics |
| `metrics_test.json`, `predictions_test.csv` | test metrics and per-utterance probabilities, read only after a selection file exists |
| `predictions_val.csv` | per-utterance validation probabilities |
| `efficiency.json` | parameter count, inference time per utterance, and spike rates for the SNNs |
| checkpoint, `log.txt` | best epoch weights, full console output |

Tables are built by reading run folders. Nothing appends to a shared results file. Each table and figure has one builder, which writes a small file of the numbers it drew.

### 6.6 Tests that run before any overnight job

- Metric code against hand-computed answers (UA, WA, F1 on small cases).
- The label parser against the Phase 0 index: all 10,039 rows identical.
- Our LIF layer against a plain reference implementation on fixed inputs.
- Settings validation: a missing, extra or mistyped field must raise. That includes a number that arrives as text, such as `64# comment`, the fault that stopped the summer run (6.7).
- The statistics code on cases with known answers, including a paired bootstrap whose pairing a wrong implementation would break.
- Layout: no .py file outside `src/` and `tests/`, and no module-level settings in the package.
- A two-batch smoke run of every model on CPU and on each GPU.

### 6.7 What the summer project's code taught

Read 2026-09-25 from `C:\Users\Username\Downloads\small`, which you named as the latest node copy: the 59 .py files at its root, the 11 shell scripts, `INVENTORY.txt`, `README_FIRST.md`, `configs/default.yaml`, `src/`, the notebooks' setup cells, and the logs. Each row names its evidence and the rule it becomes here.

| What happened | Evidence | Rule in this build |
|---|---|---|
| Changing one setting meant patching scripts and moving their output | `redirect_output.py` rewrote the output line of 4 scripts to `findings_2140` and seeded the new folder with copies, because "two scripts read a file, append to it and write it back". `nsamples=200` was edited to 2140 in the same 4 scripts, leaving `.bak200` copies | A changed setting is a new sweep with its own folder. No code picks its output path, and nothing reads, appends and writes back |
| Settings scattered as constants | `SEED` defined at module level in 21 root scripts, `IG_STEPS` in 15, `CAP` in 11 | Settings exist only in the validated settings object |
| Logic copied between scripts | 33 function names defined in two or more root scripts, e.g. `gini` in 4, `load` in 5 | Shared logic lives once in `src/affect`; the layout test blocks new root scripts |
| The package stopped growing while scripts took over | `src/` files dated Aug 1, two on Aug 15; root scripts kept arriving until Sep 17 (`bootstrap_ci.py`) | Everything runs through one CLI; a new analysis is a registered module writing to `artifacts/analyses/` |
| Several scripts wrote the same result file | `stability_full.json` written by `run_final.py` and `stability_full.py`; `frozen_validation.json` by `bulletproof.py`, `fill_gaps.py` and `weekend_run.py` | A job writes only its own folder; tables are built at read time |
| A config edit turned numbers into text | `apply_sweep_results.py` rewrote the YAML with a regex; `redo.sh`: "two numbers were read as text and every model built from them failed at its first layer" | No program edits settings files; types are checked when settings are validated |
| Config, code defaults and saved models disagreed | `confirm_settings.py` exists to state settings "from the artefacts rather than from the config or the source defaults, which disagree"; `train_cnn_bilstm` defaults `conv_channels=64` while the config says 128 | No defaults in the package; the settings shown beside a number come from that run's `spec.json` |
| The MPS fix was sometimes skipped | `run_final.py` and `gnn_last.py` import torch before `src.gpu_setup`; `confirm_gaps.py` and `confirm_settings.py` never import it; `logs/ks_f.log` and `logs/ks_h.log` open with "Error 805: MPS client failed to connect to the MPS control daemon" | The fix runs inside the package's own import, the worker sets the same variables before Python starts, and a run refuses to start without the expected GPUs (8.3) |
| GPU chosen inside a notebook | `03a_cnn_bilstm_run.ipynb` sets `CUDA_VISIBLE_DEVICES` to "1" | Only the worker assigns GPUs |
| Notebooks rewritten by runs and by scripts | `lane.sh` and `run_night.sh` run `nbconvert --execute --inplace`; `README_FIRST.md` needed an overlay plus hand edits to four notebooks; `patch_harvest_all.py`, `patch_save_findings.py` and `fix_harvest_cell.py` inserted or repaired cells | Runs write new dated copies; nothing edits a source notebook but you |
| Whole-tree copies per variant | `README_FIRST.md` of the summer project: copy the whole project folder to a new one and swap in a config that differs only in the split ratio; 13 such copies sat in `Downloads` | One tree; a variant is a setting |
| Stored results too coarse for a paired test | `agreement_paired.py`: only means per seed were kept, leaving "six numbers per comparison" | Every run keeps per-utterance predictions |
| A bootstrap bug collapsed every interval | `README_FIRST.md`: drawing a separate resample per array "broke the pairing between labels and predictions" | The statistics module is tested on known answers, pairing included (6.6) |
| A settings asymmetry crept back in | `tonight.sh`: KernelSHAP ran on forty items per class where every other method had a hundred and fifty | Settings shared across models are set once, and differ only when a sweep varies them on purpose |
| Models used different splits | `patch_gnn_splits.py` and `derive_split_indices.py` exist because the graph model drew its own permutation | Folds are defined once by the dataset module, saved with a hash, and read by every model |

What worked, and carries over: one lane per GPU with no shared output files (`lane.sh`), barriers so a stage never starts on half-finished results (now gates, 5.1), a preflight check before a long run (`preflight.py`, now `env-check`), a one-command morning report (`check_morning.sh`, now `status`), ordering work by value (`finish.sh`, now priority), and the thread caps (8.3).

---

## 7. Notebooks

### 7.1 How they run

As in your last project: notebooks run headless on the node, started by a shell script inside tmux, with the output streaming to the terminal and to a log file. The old command, from `lane.sh` and `run_night.sh`, was `python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 <notebook>`. The `--inplace` part wrote the outputs back into the source notebook, which is part of why the notebooks grew to hundreds of kilobytes and needed overlays (6.7). Plan: papermill, which runs the notebook with a mode passed in and writes the result to a new file.

| Script | What it does | Where the output goes |
|---|---|---|
| `enqueue.sh <notebook>` | runs the notebook in "enqueue" mode: checks the settings, writes the sweep's jobs into the queue | the queue, plus `artifacts/sweeps/<sweep_id>/` |
| `worker.sh <gpu>` (4 tmux sessions) | trains queued jobs on one GPU | run folders, plus `logs/worker_gpu<k>.log` |
| `report.sh <notebook>` | runs the notebook in "report" mode after a sweep: every table and figure | a new dated copy in `artifacts/notebooks/`, plus an HTML export |

The source notebook is never overwritten; each report is a new dated copy, so the notebook you present is the one that ran. Training runs in the worker processes rather than inside a notebook kernel, so one crashed job doesn't take the rest of the night with it, and each GPU has its own log. For a single quick configuration, a notebook can still call the runner directly and train in place.

### 7.2 What a model notebook holds

Thin, as you asked. A model notebook holds:

1. One import line.
2. The SETTINGS cell: every data, feature, model, training and evaluation value, one line each with a comment. This is what your professor reads and edits.
3. A check-and-show call: validates the settings and prints them as a table, plus the full layer table of the built model with parameter counts.
4. A sweep cell: the factors and values to test, and one call that writes the jobs and the sweep record.
5. Results calls: each renders one table or figure from artifacts, with the settings read from the runs' own `spec.json` files.
6. Optional: a call that shows a model's source code from the package (`inspect.getsource`), so the code is on screen in a meeting without a second copy that could go stale.

`10_results.ipynb` holds the cross-model figures. The package does all the work, so adding a model adds one small notebook and adding a dataset adds none.

---

## 8. Data assembly, node and sync

### 8.1 Assembled dataset

Built 2026-09-25 (VERIFICATION_LOG entries 3 to 9). The three received folders were copied into `data\iemocap\raw\` with a manifest per folder (path, bytes, SHA-256 of every file; each copy read back and compared, and each received folder listed again afterwards to confirm nothing changed). `python -m affect.cli assemble` built `data\iemocap\assembled\` from the received release folder, the collaborator's wav folder and the Phase 0 audit folder, each named on the command line and recorded:

| Part | Files |
|---|---|
| sentence audio, `audio/Session<n>/<dialog>/<utterance>.wav` | 10,039 (501 from his folder) |
| label summary files, `labels/` | 151 |
| transcripts, `transcripts/` | 151 |
| the Phase 0 index, `reference/phase0_iemocap_index.csv` | 1 |
| all, 1,440,743,852 bytes | 10,342 |

It refuses to finish unless every check passes, and they all did: every wav is mono 16-bit 16 kHz PCM, every duration is within 0.15 ms of its label times, the parsed labels match the Phase 0 index on every field of all 10,039 rows, the four-class counts match 1.1, his 501 fills carry the hashes from his ingested folder, and all 1,669 of his files that duplicate non-empty release files are identical to them. `MANIFEST.csv` lists every file, and its own SHA-256 (d6c6aefcc5de...) is the data version stamped into every run. `verify-data` then re-read all 10,342 files independently and repeated every check, twice: the second time, after the release copy's manifest existed, it also confirmed that all 9,840 files taken from the release carry the same hashes as the release copy in `raw\`. All passed. This copy is what goes to the node. The received folders were only read.

### 8.2 One tree, synced by direction

Code flows up (`src/`, `notebooks/`, `scripts/`, `tests/`), results flow down (`artifacts/`), and neither side edits the other's half. Two copies of the same tree is how a stale copy gets read, so there is one tree, separated by direction. Code is never edited on the node; a fix made there at 2 am is copied back before anything else the next day. The queue lives only on the node, because `enqueue.sh` runs there.

You move files with scp, run from your PC. Copying folders with `scp -r` has one trap: it overwrites files but never deletes them, so a module removed on your PC would linger on the node and could still be imported. So each direction moves one archive and checks it:

| Step | Where | What it does |
|---|---|---|
| `push.ps1` | your PC | packs the code into one archive with a hash of its contents, and copies it up with scp |
| `unpack.sh` | node | replaces the node's code folders with the archive's contents, so nothing stale survives, and records the hash |
| `pack.sh <sweep>` | node | packs a finished sweep's results into one archive with a manifest of file hashes |
| `pull.ps1 <sweep>` | your PC | copies the archive down with scp, unpacks it into `artifacts/`, and checks every hash against the manifest |

Every run records the code hash, so each result traces to the exact code that made it, with or without git. Git on your PC is still worth having for history. IEMOCAP never goes into git (size, and it is licensed data).

### 8.3 Node environment

From your node check on 2026-09-24 (live), plus the summer project's files where noted:

| Item | Value |
|---|---|
| Host | `seas-gpu-node-01`, reached with ssh under each person's own login |
| GPUs | 4 NVIDIA H100 NVL, 95,830 MiB each, driver 610.57.04; 70 MiB used on each at the check, so idle |
| The daemon | `nvidia-cuda-mps-control` and `nvidia-cuda-mps-server` are running. Without the fix, torch counts 4 devices but reports CUDA unavailable, with "Error 805: MPS client failed to connect" |
| The fix | `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` set to empty strings before torch loads (`src/gpu_setup.py` in the summer project). Verified live: with the fix, CUDA is available on all 4 |
| Thread caps | `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` set to 32, because OpenBLAS there "is built for fewer threads than the box has cores" (`lane.sh`) |
| Python | 3.12.13, built with Red Hat GCC 14.3.1 |
| Old environment | an environment from Harrison's earlier project: torch 2.6.0+cu124, nbconvert 7.17.1, numpy 2.4.6, pandas 3.0.3, scipy 1.18.0, scikit-learn 1.9.0. It has no torchaudio, transformers, papermill, librosa, soundfile or SNN library |
| Storage | `/home` is a network mount, 4.0 TB with 2.9 TB free, but each home folder has a limit of about 95 GB (Harrison's showed 40 GB free at the checks) |
| Network | PyPI and Hugging Face both answer (HTTP/2 200), so packages and model weights can be downloaded on the node |
| Operating system | AlmaLinux 10.2 |
| CPU and memory | 256 CPU threads; 1,511 GiB of RAM, 1,484 GiB available at the check; `/dev/shm` 756 GB |
| Local disk | `/scratch` is its own local XFS volume: 499 GB, 337 GB free, writable by everyone (`drwxrwxrwx`, no sticky bit), with no folder of yours yet. The root filesystem is local XFS with 316 GB free; `/tmp` and `/var/tmp` live on it and are emptied of files older than 10 and 30 days (`q /tmp ... 10d`, `q /var/tmp ... 30d`). `/data` is a root-owned folder on the root filesystem that you cannot write to |
| GPU state | compute mode Default and MIG off on all 4; no processes on any GPU at the check |
| Tools | tmux (version string `next-3.4`) and git are installed; `nvcc` is not on the path, which is fine because torch's pip wheels bring their own CUDA libraries |
| Transfers | scp from your PC, both directions (8.2) |

**New environment.** Agreed: a fresh Python environment for this project, outside the project folder, at `~/venvs/affect` (not `/scratch`, decided 2026-09-25), built from the system Python `/usr/bin/python3.12` (3.12.13) by `scripts/setup_env.sh` with the packages in `requirements.in`. Pip keeps no download cache, to spare the home quota. Two facts shape it:

- PyTorch's current stable release is 2.13, and its default wheels use CUDA 13.0 (PyTorch release page, checked 2026-09-25). Whether driver 610.57.04 runs them is checked by `env-check`, not assumed.
- torchaudio is in maintenance mode. Its maintainers moved audio loading and saving to TorchCodec as of version 2.9, while "most of the APIs in transforms, functional ... will remain". So wav files are read with the standard library's `wave` module, as Phase 0 did (every file is plain 16-bit PCM, checked for all 10,039), and log-mel is computed in torch and checked against librosa in a unit test.

The exact versions installed are frozen into `artifacts/env/pip_freeze_<utc>.txt` when the environment is created, and every run records them. Packages for later steps (transformers, the SNN library) are added when their step starts.

The fix is applied three ways, so one missed import can't bring back the silent failure: `scripts/common.sh` exports the variables before Python starts (every node script sources it), the package sets them in its own `__init__.py` and refuses to import if torch was loaded first without them, and a run checks that the expected GPUs are visible and refuses to fall back to the CPU. `common.sh` also holds the thread caps. `python -m affect.cli env-check` records the Python, torch, CUDA and driver versions, the GPU names and free memory, whether CUDA is visible after the fix, whether the node can download packages and model weights, and a training step on each GPU. The output is saved in `artifacts/checks/`, and the pilot starts only if it passes. `scripts/check.sh` runs the tests, `verify-data` and `env-check` in that order and stops at the first failure. Training refuses to run on the CPU unless told to, so a missed import fails loudly.

### 8.4 Storage budget

Memory is plentiful and home storage is not, so the budget leans on RAM:

- **Audio and features in memory.** Each worker loads the assembled audio once (about 1.4 GB as 16-bit samples) and computes log-mel on the GPU per batch. With 1,511 GiB of RAM, even every feature variant, or a frozen backbone's per-layer features for all four-class utterances (tens of GB at half precision, by rough arithmetic; measured when built), can sit in memory. The workers state their memory use in the log, because the node is shared.
- **Disk only for results.** Run folders hold settings, metrics, predictions and logs: small files.
- **Checkpoints by stage.** Screening runs keep none; their predictions and metrics are enough for selection and statistics. Backed and final runs keep checkpoints for the small models. For the large pretrained models, only the trained part (the head, or the fine-tuned layers) is kept, in bf16, and only for final configurations.
- **Where things live on the node.** Nothing goes in `/scratch` (your decision, 2026-09-25), so everything shares the home folder's limit of about 95 GB, 40 GB free at the checks on 2026-09-24 and 25.

| Place | What | Size |
|---|---|---|
| `~/affect/` | code (replaced whole by each push), `data/iemocap/assembled/`, `artifacts/`, `queue/`, `logs/` | code small; data 1.44 GB (8.1); results grow per sweep |
| `~/venvs/affect/` | the environment | measured by `setup_env.sh` when built; not known yet |
| `~/cache/huggingface/` | pretrained weights, from step 5 | measured when downloaded |
| `~/affect_incoming/`, `~/affect_outgoing/` | archives moving up and down | code archives are small; the data archive is 1.44 GB once |
| never `/tmp` or `/var/tmp` | nothing we keep | emptied on a timer |

  The environment variables that point here (`HF_HOME` and the environment path) are set in one place, `scripts/common.sh`, next to the MPS fix. The environment can be rebuilt from `requirements.in` and the frozen versions.

- **All the raw data on the node?** The full release is 23,987,105,213 bytes (24.0 GB) in 83,151 files (VERIFICATION_LOG entry 3). It would fit in 40 GB only by leaving about 16 GB for the environment, the model weights and every result, against the 10 GB guard below, before the environment's size is even known. And nothing planned reads most of it: motion capture is 18.4 GB and video 1.47 GB (Session 1 only). So only the assembled copy goes up. If a later model needs the dialog recordings (2.67 GB) or motion capture, that part gets its own assembled copy then. If the whole release were ever wanted on the node, the room would have to come from other folders in that home folder.
- **A guard.** A worker won't start a job when free space where it writes is below a set floor (proposed 10 GB), and `status` reports disk use every morning.
- **Pruning after transfer.** Once `pull.ps1` has verified a sweep's archive on your PC, the node copy of that sweep can be deleted.

---

## 9. Real time, and the chatbot later

A spectrogram is not what makes a model slow. Computing a log-mel spectrogram costs about one short FFT per 10 ms of audio, and most speech systems compute one as their first step; Whisper's pipeline does. Delay comes from two things: how much audio the model needs before deciding (emotion is usually judged per utterance, a few seconds) and model size. A small CNN on three seconds of log-mel costs far less than WavLM Large.

Your transcript idea matches how such systems are usually built. Voice activity detection cuts the audio into turns, speech recognition (for example Whisper) produces the words, the audio emotion model runs on the same turn, and the chatbot gets both. Choices now that keep that path open:

- 16 kHz mono input, with feature extraction inside the model code, so training and live audio share one code path.
- Parameter count and inference time per utterance recorded for every model from the first run (`efficiency.json`).
- Improvised-only results kept, since they are closer to spontaneous speech.
- IEMOCAP is ten actors performing. A therapy session is not. The second dataset will show how much transfers, and EmoBox defines cross-corpus tests (IEMOCAP, MELD, RAVDESS, SAVEE) to compare against.

---

## 10. Terminology

| Term | Meaning |
|---|---|
| utterance | one wav in `sentences/wav` and its label line |
| dialog | one conversation, e.g. `Ses01F_impro01` |
| session | one of five recording sessions, two actors each |
| speaker | session plus the letter in the utterance's last field; 10 in total |
| fold | one test session under leave one session out |
| validation session | the training session held out for early stopping and selection |
| screening run | one seed; picks what to test next; never reported as a finding |
| backed result | at least 5 seeds with spread and interval; can be reported |
| UA / WA | mean per-class recall / overall accuracy |

The project was renamed from Effective to Affective Computing on 2026-09-25, the usual spelling of the field.

---

## 11. Do not assume

- The letter after `SesXX` is not the speaker. The letter in the last field is.
- Your IEMOCAP copy is not complete: 501 Session 5 sentence wavs are empty. Use the assembled dataset.
- The collaborator's spectrograms for Session 5 show the wrong utterances. His Sessions 1 to 4 are right.
- The per-evaluator `Attribute` files reverse valence and activation. Use the summary files.
- The release README says `Evaluation/`; the folder is `EmoEvaluation/`.
- A random split never produces a reportable number.
- Published IEMOCAP numbers compare only when class set, split and subset all match.
- A pretrained checkpoint may have seen IEMOCAP.
- A single-seed result is a screening result, not a finding.
- Test results are read only after a selection file exists.
- Normalization statistics come from training sessions only.
- On the node, importing torch before the MPS fix breaks CUDA, and a script can carry on without the GPU. A run that can't see its GPUs must stop.
- `nbconvert --inplace` overwrites the notebook it runs. Runs write new dated copies.
- `/home` on the node is a shared network drive with limited free space. Nothing large goes there without a place in the storage budget (8.4).
- `scp -r` never deletes. Code moves as one archive that replaces the node's copy (8.2).
- Current torchaudio no longer provides audio loading. Wav files are read with the standard library's `wave` module.
- `/tmp` and `/var/tmp` on the node are emptied of files older than 10 and 30 days. Nothing we keep goes there.
- Nothing in this project uses `/scratch` (decided 2026-09-25). The environment, the model cache and the data are in your home folder.
- Your home folder on the node has a limit of about 95 GB even though `/home` shows 4.0 TB.
- A variant is a setting, not a new folder or a new script.
- The release has video for Session 1 only, and motion capture is incomplete. Don't plan a multimodal model on them without checking coverage first.
- The PC's shared-folder bridge (the Cowork shell) is slow, and one command may run only a few minutes: copying the 24.0 GB release took 51 resumable calls over about three hours (VERIFICATION_LOG entry 6). Anything large on the PC is done in resumable steps (`ingest`, `assemble` and `verify-data` all resume).
- A file committed to the PC through the bridge was once still read as the old version by the next command. Check a file's hash through the PC shell after committing it, before running it.

---

## 12. Decisions

Answered 2026-09-24 and 2026-09-25:

- Use his verified copies of the 501 files; regenerate his image format instead of training on his files.
- Notebooks run headless on the node from shell scripts.
- Node: host `seas-gpu-node-01`, login by ssh, files moved by scp from your PC, and a fresh Python environment for this project (results of the checks in 8.3).
- 2.6's plan to keep rebuildable things in `/scratch` is withdrawn: nothing uses `/scratch` (2026-09-25). Everything is in your home folder (8.4).
- The project is Affective Computing; the repository and package are `affect` (2026-09-25).
- Only the assembled training copy goes to the node, not the whole release (8.4, 2026-09-25).

Defaults in use until you change them:

1. The new environment at `~/venvs/affect`, rebuilt from `requirements.in` and the frozen versions if lost.
2. Seeds: 5 for every reported number. For the paper's head-to-head claims, more seeds until the comparison is clear, up to 29 (Bouthillier et al., 3.3).
3. Screening: one seed, moving to three if the pilot's seed-to-seed SD of fold-mean UA is above 1.0 point (3.4).
4. The repository at `C:\Affective Computing\affect\` on your PC and `~/affect/` on the node.

Still open: the SNN library, SpikingJelly or snnTorch. It is chosen on the node once the new environment exists, by a speed test and the reference-implementation check (4.4).

---

## 13. Build order

1. Repository skeleton with the settings validation, registries, run folders, and tests (6.6). Done 2026-09-25: 72 tests; on the PC all pass except one that needs scipy, which the PC shell lacks and the node has.
2. Data: the IEMOCAP module, the assembly command, and the index parity check. Done 2026-09-25 (8.1).
3. The new environment on the node, then `env.py` and `env-check`, then the check on all 4 GPUs. Next: `push.ps1 -Data`, `unpack.sh`, `setup_env.sh`, `check.sh` (HANDOFF.md). `env.py` and `env-check` are written; they first run on the node.
4. CNN, the controls, the pilot night, then CNN screening and seeded comparisons.
5. CNN + Transformer, CNN + LIF (both variants), then the pretrained model, each through the same stages.

---

## Sources opened for this plan

- Phase 0 audit, this computer: `C:\Affective Computing\_phase0_audit\README.md` and the JSON files beside it
- IEMOCAP release `README.txt`, this computer: `C:\Affective Computing\IEMOCAP_full_release\README.txt`
- Build records, this computer: `C:\Affective Computing\affect\VERIFICATION_LOG.md`, `data\iemocap\assembled\ASSEMBLY.json`, `data\iemocap\manifests\`
- Busso et al. 2008, IEMOCAP: https://biic.ee.nthu.edu.tw/archive/doc/research/IEMOCAP%20interactive%20emotional%20dyadic%20motion%20capture%20database.pdf
- EmoBox, Ma et al. 2024: https://arxiv.org/html/2406.07162
- EmoBox IEMOCAP label map: https://raw.githubusercontent.com/emo-box/EmoBox/main/data/iemocap/label_map.json
- emotion2vec, Ma et al. 2024: https://arxiv.org/html/2312.15185v1
- emotion2vec+ model card: https://huggingface.co/emotion2vec/emotion2vec_plus_large
- WavLM, Chen et al. 2022 (pretraining data only): https://arxiv.org/pdf/2110.13900
- SpeechBrain IEMOCAP model card: https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP
- Antoniou et al. 2023: https://ar5iv.labs.arxiv.org/html/2304.00860
- Wang et al. 2021: https://ar5iv.labs.arxiv.org/html/2111.02735
- Padi et al. 2021: https://arxiv.org/html/2108.02510v2
- Neumann and Vu 2017: https://www.isca-archive.org/interspeech_2017/neumann17_interspeech.pdf
- Bouthillier et al. 2021, Accounting for Variance in Machine Learning Benchmarks: https://arxiv.org/pdf/2103.03098
- Cawley and Talbot 2010, On Over-fitting in Model Selection: https://www.jmlr.org/papers/v11/cawley10a.html
- Dror et al. 2019, Deep Dominance: https://aclanthology.org/P19-1266/
- Bittar and Garner 2022, surrogate gradient spiking baseline for speech commands: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.865897/full
- SpikingJelly Triton backend benchmark: https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/triton_backend.html
- Spikformer (energy estimation): https://openreview.net/pdf?id=frE4fUwz_h
- snnTorch `Leaky`: https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html
