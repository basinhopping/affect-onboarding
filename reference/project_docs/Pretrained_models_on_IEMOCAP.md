# Pretrained speech models used on IEMOCAP: what papers report (checked 2026-10-12)

Written 2026-10-12 (early morning UTC) for the meeting of 2026-10-12 and the HuBERT work after it. Every figure below was read on the source page itself (arXiv PDF or HTML, Hugging Face card) and then checked a second time, claim by claim, by a separate agent the same night: 17 claims, none contradicted, two details not stated in the sources (marked below). Figures are as printed; the meeting doc rounds them half up to one decimal.

All rows: IEMOCAP, four classes (neutral, happy with excited merged in, angry, sad). The merge is stated in emotion2vec, Pepino, Chen and Rudnicky and Morais; SUPERB's paper says only "four classes" and does the merge in s3prl's IEMOCAP_preprocess.py; EmoBox's paper lists 5 emotions and its repo's label_map.json maps exc to Happy; Wang et al. say "4 emotional classes as in SUPERB".

## Frozen models (only a small classifier is trained)

| Model | Params | Pretraining audio | Protocol | Result | Source |
|---|---|---|---|---|---|
| wav2vec 2.0 Base | 95.04M | LibriSpeech 960 h | weighted sum of layers, 5 folds by session | ACC 63.43 | SUPERB Table 2 (arXiv 2105.01051) |
| HuBERT Base | 94.68M | LibriSpeech 960 h | same | ACC 64.92 | SUPERB Table 2 |
| wav2vec 2.0 Large | 317.38M | Libri-Light 60k h | same | ACC 65.64 | SUPERB Table 2 |
| HuBERT Large | 316.61M | Libri-Light 60k h | same | ACC 67.62 | SUPERB Table 2 |
| WavLM Base | 94.70M | LibriSpeech 960 h | SUPERB | ACC 65.94 | WavLM Table I (arXiv 2110.13900) |
| WavLM Base+ | 94.70M | 94k h (Libri-Light 60k, GigaSpeech 10k, VoxPopuli 24k) | SUPERB | ACC 68.65 | WavLM Table I |
| WavLM Large | 316.62M | 94k h | SUPERB | ACC 70.62 | WavLM Table I |
| UniSpeech-SAT Large | 316.61M | 94k h | SUPERB | ACC 70.68 | UniSpeech-SAT Table 1 (arXiv 2110.05752) |
| data2vec (base) | 93.75M | LibriSpeech 960 h | session out, 20% of training as validation | WA 67.38 | emotion2vec Table 2 (arXiv 2312.15185) |
| emotion2vec | 93.79M | LS-960 plus 262 h emotional speech (IEMOCAP, MELD, MEAD, CMU-MOSEI, MSP-Podcast) | same | WA 71.79, UA 72.69 | emotion2vec Table 2; UA in Tables 8 to 10 |
| Whisper large-v3 | 1550M (whole model) | 1M h weakly labeled + 4M h pseudo-labeled | last Transformer layer features, 5 folds of 2 speakers | UA 73.54, WA 72.86 | EmoBox Table 4 (arXiv 2406.07162) |
| HuBERT Large | 316.61M | Libri-Light 60k h | same as above | UA 67.42 | EmoBox Table 4 |
| WavLM Large | 316.62M | 94k h | same as above | UA 69.47 | EmoBox Table 4 |
| wav2vec 2.0 Base | 95M | LibriSpeech 960 h | all layers (trainable weighted average), Dense head, session out, one training session for early stopping | UA 67.2 ± 0.7 (speaker normalization), 65.8 ± 0.3 (global) | Pepino et al. 2021 Tables 1 and 2 (arXiv 2104.03502) |

## Fine-tuned models

| Model | Protocol | Result | Source |
|---|---|---|---|
| wav2vec2-base | session out, all four other sessions, a fixed 15 epochs without validation, 5 runs | UA 69.9 (V-FT), 73.5 (TAPT), 74.3 (P-TAPT) | Chen and Rudnicky 2023 Table 1 (arXiv 2110.06309) |
| HuBERT (size never stated) | session out, validation a random 20% of the 8 training speakers' samples | WACC 72.99 / UACC 73.45; with checkpoint averaging (5 best by validation) 75.20 / 75.80 | Morais et al. 2022 Table 1, rows 2 and 4 (arXiv 2202.03896) |
| HuBERT Large, partly fine-tuned (CNN encoder frozen) | 10 folds, leave two speakers out (one validation, one test) | WA 73.01; 79.58 with random splits (speakers shared) | Wang et al. 2021 Table 1, PF-hbt-large (arXiv 2111.02735) |

## Cautions (all in the meeting doc)

- emotion2vec pretrains on IEMOCAP's audio (Table 1 ticks IEMOCAP under Pretrain and Downstream); the paper never discusses it.
- EmoBox: "The optimal settings, determined by the best performance outcomes, are then selected"; no split named. The word "frozen" never appears; features are extracted from the last layer.
- Pepino's speaker normalization uses all of a speaker's audio, test speakers included.
- Chen and Rudnicky: no validation set; TAPT and P-TAPT continue pretraining on "the target dataset", without saying whether the held-out session is excluded.
- Wang et al.'s 79.58 and SpeechBrain's model card (78.7%, average 75.3%; its recipe's default `different_speakers: False`, `split_ratio: [80, 10, 10]`, a random split) share speakers between training and test, like the project's pooled split.
- emotion2vec's own Table 2 prints other figures for WavLM (WavLM+ 67.98, WavLM Large 70.03): cite WavLM's table for WavLM.
- emotion2vec Table 11's α = 1 row prints WA 72.14 / UA 72.84, not the 71.79 / 72.69 of the other ablation tables; the paper does not reconcile them.

## Checkpoints (Hugging Face)

- facebook/hubert-base-ls960 (95M) and facebook/hubert-large-ll60k (317M): apache-2.0. facebook/hubert-xlarge-ll60k (964M).
- microsoft/wavlm-base, wavlm-base-plus, wavlm-large (94k h): license link to microsoft/UniSpeech, CC BY-SA 3.0.
- facebook/wav2vec2-base (apache-2.0), facebook/wav2vec2-large-lv60, facebook/xls-r-300m.
- facebook/data2vec-audio-base and -large (apache-2.0).
- openai/whisper-large-v3 (apache-2.0, 1550M).
- emotion2vec/emotion2vec_base and emotion2vec_plus_large (license "other", FunASR model license).
- audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (CC BY-NC-SA 4.0; arousal, dominance, valence from MSP-Podcast).
- superb/hubert-large-superb-er (apache-2.0; the SUPERB ER head).
- speechbrain/emotion-recognition-wav2vec2-IEMOCAP (apache-2.0; random split by default, see above).

## For the project

cnn_final_3 has 1,077,988 parameters and the CNN + Transformer 947,668 (each run's efficiency.json); HuBERT Base is about 88 times cnn_final_3. The meeting doc suggests a first HuBERT run the SUPERB way (frozen, weighted sum of layers, small classifier) on the project's exact folds and seeds, so it pairs with the CNN results, and fine-tuning after that.
