# Code walkthrough (private)

Oct 11, 2026 · @Harrison

This walks the code Dr. Luitel is most likely to ask about, the models and their training, block by block with the line numbers of the GitHub commit d193f4e, so each link opens the exact lines. Every other file is explained by what it does at the end.

## How one experiment flows through the code

&#91;embedded content: experiment flow · 9 steps, sweep file to test report\]

The left column ends in one run folder per configuration, fold and seed; the right column compares them on validation, records the choice, and only then reads its test results.

## The input: log-mel features and SpecAugment

Every neural model reads a log-mel spectrogram computed on the GPU from 16 kHz audio, then normalized and, in training only, masked by SpecAugment.

### `src/affect/features/logmel.py`

**[Lines 1 to 7](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L1-L7): the module docstring.** The log-mel is computed in torch on the job's device, one batch at a time, so features are rebuilt from the waveform for every batch. The mel filter bank copies librosa's default: the Slaney mel scale with Slaney area normalization. Frames are centered and zero padded, so an utterance of n samples gives `1 + n // hop` frames.

**[Lines 8 to 13](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L8-L13): imports.** The top imports are the standard library, numpy and the registry; torch is imported inside the class (lines 98 and 114). That lets `check` run where torch is missing, such as when a sweep is enqueued.

**[Lines 16 to 24](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L16-L24): the numeric settings.** `LogMelSettings` is a frozen dataclass with no defaults, so every job must state every field. `win_ms` and `hop_ms` are the window length and the step between windows, in milliseconds. `n_fft` is the FFT size in samples, `n_mels` the number of mel bands, and `f_min` to `f_max` their range in Hz. `log_offset` is "added to the mel power before the log, so silence stays finite".

**[Lines 25 to 31](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L25-L31): the normalization choice.** `normalization` names one of four modes, defined in the comments and applied in `train/loop.py` (next section). "utterance" scales each mel band to mean 0 and SD 1 over that utterance's own frames. "global" uses one mean and SD per band from all training-session frames; "speaker" uses one per speaker, from all of that speaker's audio. "none" keeps the raw log-mel, so loudness survives (the comment calls it "an arousal cue").

**[Lines 34 to 38](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L34-L38): the Slaney mel-scale constants.** The mel scale is a frequency scale that follows perceived pitch, so it gives low frequencies more bands than high ones. Below 1000 Hz it is linear, at 200/3 (about 66.7) Hz per mel, so 1000 Hz is 15 mels. Above 1000 Hz it is logarithmic: every 27 mels multiply the frequency by 6.4. Line 38 returns these numbers: 200/3, 1000, 15 and ln(6.4)/27.

```latex
m(f) =
\begin{cases}
\dfrac{f}{200/3}, & f < 1000 \\[6pt]
15 + \dfrac{\ln(f/1000)}{\ln(6.4)/27}, & f \ge 1000
\end{cases}
\qquad (f \text{ in Hz})
```

**[Lines 41 to 45](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L41-L45): Hz to mels.** `hz_to_mel` applies the formula above to a whole array with `np.where`. `np.where` computes both branches, so `np.maximum(freq, 1e-12)` keeps the log from ever seeing 0 Hz.

**[Lines 48 to 52](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L48-L52): mels to Hz.** `mel_to_hz` is the exact inverse. It turns evenly spaced mel points back into Hz.

**[Lines 55 to 59](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L55-L59): FFT bin frequencies and band edges.** Line 57 lists the 257 frequencies of a 512-point FFT at 16000 Hz: 0, 31.25, 62.5 and so on up to 8000 Hz. Line 58 places `n_mels + 2` points evenly on the mel scale from `f_min` to `f_max`, then converts them to Hz. Equal mel steps are narrow in Hz at low frequencies and wide at high ones. With 40 bands, 13 band peaks lie below 1000 Hz; the lowest band spans 0 to 147.1 Hz and the highest 6873.7 to 8000 Hz.

```latex
f_j = m^{-1}\!\left( m(f_{\min}) + \frac{j}{n_{\mathrm{mels}} + 1}\,\bigl( m(f_{\max}) - m(f_{\min}) \bigr) \right),
\qquad j = 0, 1, \dots, n_{\mathrm{mels}} + 1
```

**[Lines 60 to 65](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L60-L65): the triangular filters.** `ramps` holds the signed distance in Hz between every edge point and every FFT bin, and `fdiff` (line 59) the gaps between points. For band i, `lower` climbs from 0 to 1 between points i and i + 1; `upper` falls back to 0 at point i + 2. The weight is the smaller of the two, floored at 0, which draws a triangle. Each band therefore sums the power of nearby FFT bins, weighting its center most.

**[Lines 66 to 67](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L66-L67): Slaney area normalization.** `enorm` is 2 divided by each band's width in Hz, which gives every triangle an area of 1. Without it, a wide high band would add up many more FFT bins than a narrow low band. The function returns a `[n_mels, 257]` matrix of weights.

```latex
W_{i,k} = \frac{2}{f_{i+2} - f_i}\,
\max\!\left(0,\ \min\!\left( \frac{\varphi_k - f_i}{f_{i+1} - f_i},\ \frac{f_{i+2} - \varphi_k}{f_{i+2} - f_{i+1}} \right) \right),
\qquad \varphi_k = 31.25\,k \text{ Hz},\ k = 0, \dots, 256
```

**[Lines 70 to 72](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L70-L72): milliseconds to samples.** `window_samples` rounds `ms * sample_rate / 1000` to whole samples. At 16000 Hz, `win_ms` 25.0 gives 400 samples, `hop_ms` 10.0 gives 160 and `hop_ms` 20.0 gives 320. With a 160-sample hop, neighboring windows overlap by 240 samples (15 ms).

**[Lines 75 to 85](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L75-L85): refusing settings that cannot work.** `check` raises if the window or hop rounds to less than one sample, or if the window is longer than `n_fft`. It requires `0 <= f_min < f_max <= sample_rate / 2`; 8000 Hz is half of 16000, the highest frequency the audio can hold. It also requires at least one band and a positive `log_offset`.

**[Lines 86 to 90](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L86-L90): no empty bands.** `check` builds the filter bank and refuses it if any band has only zero weights, meaning it falls between two FFT bins. With 64 bands and `n_fft` 512, the lowest band (0 to 92.8 Hz) still covers two bins, 31.25 and 62.5 Hz. The same check runs when a sweep is enqueued ([runs/spec.py lines 93 to 99](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py#L93-L99)).

**[Lines 93 to 95](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L93-L95): registration and selection.** The decorator stores `LogMel` in the feature registry as "logmel", with `LogMelSettings` as its settings ([registry.py lines 30 to 41](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/registry.py#L30-L41)). [catalog.py line 9](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/catalog.py#L9) imports this module, which runs the decorator and creates the entry. A job's `features` section holds `"name": "logmel"`, and [runs/spec.py line 51](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py#L51) checks its other fields against `LogMelSettings`. [train/loop.py line 332](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L332) then builds the extractor with the data's sample rate and the job's device.

**[Lines 97 to 107](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L97-L107): setup, once per job.** The constructor runs `check`, converts window and hop to samples, and builds a periodic Hann window of 400 samples on the device. A Hann window tapers each slice smoothly to 0 at both ends, which limits leakage between FFT bins. The filter bank is built in numpy at line 106 (`check` built one at line 86 only to test it) and kept as a float32 `[n_mels, 257]` tensor on the device, so batches reuse it. The sample rate is not a setting: it is the data's own rate, which must be the same for every utterance ([train/data.py lines 71 to 73](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L71-L73)), and IEMOCAP files must be 16000 Hz ([data/iemocap.py line 87](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L87)).

**[Lines 109 to 111](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L109-L111): frame count.** `frames` returns `1 + n_samples // hop`, which matches the centered framing. The training loop uses it to know how many frames of each padded example are real ([train/loop.py line 184](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L184)). With a 160-sample hop that is 100 frames per second, so a 10.0 s crop (160000 samples) gives 1001 frames.

**[Lines 113 to 117](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L113-L117): the short-time Fourier transform.** `__call__` takes waveforms `[batch, samples]`, and `torch.stft` takes a 512-point FFT of a windowed slice every `hop` samples. `center=True` with `pad_mode="constant"` first adds 256 zeros (`n_fft // 2`) at each end, so frame t is centered on sample t × hop. The 400-sample window is padded with 56 zeros on each side to fill 512 points, so bins are 16000 / 512 = 31.25 Hz apart. The complex output has shape `[batch, 257, frames]`.

**[Line 118](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L118): the power spectrum.** Squaring the real and imaginary parts and adding them gives the power in every bin and frame. The phase is discarded.

**[Line 119](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L119): mel power.** Multiplying by the `[n_mels, 257]` filter bank turns 257 bins into `n_mels` bands for every frame. The result has shape `[batch, n_mels, frames]`.

**[Line 120](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L120): the log.** `torch.log` is the natural log, taken after adding `log_offset` (1e-06 in every record below). A silent frame or zero padding gives ln(1e-06), about -13.8, instead of minus infinity. A change of volume multiplies the power, so after the log it becomes an additive shift that normalization can remove. The whole computation is below; the tilde marks the zero-padded waveform and the bar the padded Hann window.

```latex
\begin{aligned}
X_{b,k,t} &= \sum_{j=0}^{511} \tilde{x}_b[\,t \cdot \mathrm{hop} + j\,]\; \bar{w}[j]\; e^{-2\pi i k j / 512},
\qquad k = 0, \dots, 256 \\
L_{b,m,t} &= \ln\!\Bigl( \sum_{k=0}^{256} W_{m,k}\, \lvert X_{b,k,t} \rvert^{2} + 10^{-6} \Bigr) \\
T &= 1 + \left\lfloor n / \mathrm{hop} \right\rfloor,
\qquad \text{frames per second} = 16000 / \mathrm{hop}
\end{aligned}
```

### Normalization, in `src/affect/train/loop.py`

**[Lines 49 to 59](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L49-L59): the `Normalizer`.** It applies the mode named by `features.normalization`. "global" and "speaker" carry stored means and SDs, while "utterance" and "none" carry nothing.

**[Lines 61 to 72](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L61-L72): per-utterance z-scores.** Line 67 builds a mask of each example's real frames, so padding never enters the statistics. For each band, the mean and variance are taken over real frames only, dividing by the frame count. Each value is then centered and divided by the square root of the variance plus 1e-5.

```latex
\begin{aligned}
\mu_m &= \frac{1}{T} \sum_{t=0}^{T-1} L_{m,t},
\qquad \sigma_m^2 = \frac{1}{T} \sum_{t=0}^{T-1} \bigl( L_{m,t} - \mu_m \bigr)^2 \\
Z_{m,t} &= \frac{L_{m,t} - \mu_m}{\sqrt{\sigma_m^2 + 10^{-5}}} \ \text{ for } t < T,
\qquad Z_{m,t} = 0 \ \text{ on padded frames}
\end{aligned}
```

**[Lines 73 to 82](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L73-L82): the other modes, then padding set to 0.** "global" uses one stored mean and SD per band, and "speaker" first looks up each row's speaker. Line 82 multiplies by the mask, so padded frames are exactly 0 in every mode but "none", which returns early (lines 64 to 65).

**[Lines 85 to 135](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L85-L135): where stored statistics come from.** `global_stats` and `speaker_stats` compute per-band means and SDs over real frames, in float64. `make_normalizer` computes them only for "global" and "speaker"; "utterance" needs none.

**[Lines 348 to 349](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L348-L349): the decision, once per job.** `train_job` reads `job.features.normalization` and builds a single normalizer. For "speaker" the statistics use every row, validation and test speakers included; for "global", only the training rows. The same object normalizes training batches (line 185) and evaluation batches (line 236). All five records below use "utterance".

### `src/affect/augment/specaugment.py`

**[Lines 1 to 8](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/specaugment.py#L1-L8): the docstring and import.** SpecAugment is data augmentation: it hides random stripes of the normalized log-mel, in training batches only. A frequency mask zeroes a run of adjacent mel bands in every frame; a time mask zeroes a run of adjacent frames in every band. Zero is the band mean after normalization, so a masked cell holds the band's average, not a silence value. Each example gets `freq_masks` stripes of up to `freq_mask_width` bands and `time_masks` stripes of up to `time_mask_width` frames.

**[Lines 11 to 16](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/specaugment.py#L11-L16): drawing the widths.** `_band_mask` returns a true/false mask of shape `[batch, size]` for one kind of stripe. `limit` is each example's length along that axis: `n_mels` for frequency, the real frame count for time. Line 16 draws every width uniformly from the whole numbers 0 to `max_width`, separately per example and per mask. A width of 0 is possible, so a drawn mask can be empty.

**[Lines 17 to 18](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/specaugment.py#L17-L18): drawing the start positions.** `room` is the limit minus the width, floored at 0. Each start is uniform over the whole numbers 0 to `room`, so the stripe ends inside the limit. If a time stripe is wider than the utterance, it starts at 0 and covers every real frame.

**[Lines 19 to 21](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/specaugment.py#L19-L21): from numbers to a mask.** For each position, `inside` tests whether it lies in \[start, start + width) of each mask. `any(dim=1)` merges an example's masks, so stripes may overlap. In the formula, W is the widest allowed mask and l is the limit.

```latex
\begin{aligned}
w &\sim \mathrm{Uniform}\{0, 1, \dots, W\} \\
s &\sim \mathrm{Uniform}\{0, 1, \dots, \max(0,\ l - w)\} \\
Z &\leftarrow 0 \quad \text{on positions } s, s + 1, \dots, s + w - 1
\end{aligned}
```

**[Lines 24 to 28](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/specaugment.py#L24-L28): the entry point and the off switch.** `spec_augment` takes the normalized batch `[batch, n_mels, frames]`, the real frame counts and a CPU random generator. If both mask counts are 0, it returns the input untouched, which is the baseline's case. The generator is seeded with the job seed ([train/loop.py line 328](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L328)), so a seed always gives the same masks.

**[Lines 29 to 36](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/specaugment.py#L29-L36): frequency masks.** Every example gets the same limit, `n_mels`, so a stripe can sit anywhere among the bands. `masked_fill` sets the chosen bands to 0.0 in every frame of that example. Masks are drawn only when both the count and the width are above 0. They are built on the CPU and moved to the batch's device.

**[Lines 37 to 41](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/specaugment.py#L37-L41): time masks.** Here the limit is each example's real frame count, capped at the batch's frame count. The chosen frames are set to 0.0 in every band, and the masked batch is returned. Frequency draws come first, then time draws, from the same generator.

### When masking runs, in `src/affect/train/loop.py`

**[Lines 179 to 185](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L179-L185): the training step.** `Step.loss` computes the batch's log-mel and real frame counts, then normalizes and masks it in one line (185). That line is the only call of `spec_augment` in `src/`. Lines 188 to 192 then apply mixup when `mixup_alpha` is above 0, and the model receives the input and frame counts (line 195). The features stay float32; bf16 autocast starts only at line 193, around the model.

**[Lines 383 to 398](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L383-L398): the epoch loop.** Each utterance in a batch is randomly cropped to `round(crop_s * 16000)` samples (line 364), or zero padded when shorter ([train/data.py lines 121 to 152](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L121-L152)). Samples are int16 divided by 32768, so the waveform lies in \[-1, 1). The batch then goes through `step.loss` with the mask generator, so new masks are drawn every time an utterance is used.

**[Lines 218 to 236](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L218-L236): evaluation has no augmentation.** `predict` computes features and normalizes them, with no masks and no mixup. It serves validation (lines 411 and 454), the training-set check (line 418) and test (line 462). It feeds whole utterances, padded to the longest in each batch ([train/data.py lines 155 to 166](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L155-L166)), since no record sets `eval_segment_s`. The docstring of `augment/settings.py` states the rule: "Validation and test data are never augmented."

### `src/affect/augment/settings.py`

**[Lines 1 to 10](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/settings.py#L1-L10): the docstring.** It fixes the order inside a training step: speed perturbation, then features and normalization, then SpecAugment, then mixup. Validation and test data are never augmented. `speed_factors` and `mixup_alpha` were added on 2026-09-26, and older settings that lack them mean no speed change and no mixup.

**[Lines 11 to 13](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/settings.py#L11-L13): the import.** `speed_ratio` is imported only to validate speed factors in `check`.

**[Lines 16 to 21](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/settings.py#L16-L21): the four SpecAugment fields.** `freq_masks` is the number of frequency masks per training example, and `freq_mask_width` the widest, in mel bands. `time_masks` and `time_mask_width` are the same for time, with the width in frames. A count of 0 means no masks of that kind.

**[Lines 22 to 28](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/settings.py#L22-L28): speed and mixup.** `speed_factors` lists speeds; each time a training example is used, one is drawn with equal probability and the waveform is resampled by it. `[1.0]` means no speed perturbation. `mixup_alpha` is the alpha of the Beta(alpha, alpha) distribution of mixing weights, and 0 means no mixup.

**[Line 30](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/settings.py#L30): values for older settings.** `ADDED_FIELDS` gives settings written before 2026-09-26 `speed_factors` \[1.0\] and `mixup_alpha` 0.0, and they keep their configuration hashes. The baseline record is such a case: its `augment` section has only the four SpecAugment fields.

**[Lines 33 to 48](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/settings.py#L33-L48): validation.** Counts and widths must not be negative, and `freq_mask_width` must not exceed `n_mels`. Speeds must be listed, distinct, between 0.5 and 2.0, and each an exact fraction with a denominator of at most 100. `mixup_alpha` must not be negative.

### `src/affect/augment/speed.py`

**[Lines 20 to 62](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/speed.py#L20-L62): speed perturbation, off in every record below.** A speed of 1.1 plays the utterance 10% faster and higher, like sox "speed" ([lines 1 to 6](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/speed.py#L1-L6)). `speed_ratio` ([lines 20 to 28](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/speed.py#L20-L28)) writes the speed as a fraction, and `change_speed` ([lines 31 to 41](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/speed.py#L31-L41)) resamples with `scipy.signal.resample_poly`. `draw_speeds` ([lines 56 to 62](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/speed.py#L56-L62)) picks one speed per example, applied before the crop ([train/data.py lines 144 to 152](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L144-L152)). With `speed_factors` \[1.0\], as in every record, nothing is drawn or resampled.

### `src/affect/augment/mixup.py`

**[Lines 13 to 36](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/mixup.py#L13-L36): mixup, also off in every record below.** `draw` ([lines 13 to 19](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/mixup.py#L13-L19)) gives each example a weight w from Beta(alpha, alpha) and a partner from a shuffle of the batch, possibly itself. `mix` ([lines 22 to 36](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/mixup.py#L22-L36)) computes `w * own + (1 - w) * partner` for the input and every target, with the same w. The mixed example keeps the longer frame count, since padding is 0. It runs after SpecAugment ([train/loop.py lines 188 to 192](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L188-L192)), only when `mixup_alpha` is above 0.

### What the finalists use

The table reads the `settings` field of each `artifacts/selections/<folder>/selection.json`. `.gitignore` keeps `artifacts/` out of GitHub, so these files are cited by path, not linked.

| Record | Model | Mel bands | Training crop (`crop_s`) | Frequency masks | Time masks | Speed | Mixup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `threeblock_final` | `cnn` | 40 | 10.0 s (1001 frames) | 2, up to 12 bands | 2, up to 30 frames (0.3 s) | `[1.0]`, off | `0.0`, off |
| `lif_final` | `cnn_lif` | 64 | 5.0 s (501 frames) | 2, up to 8 bands | 2, up to 20 frames (0.2 s) | `[1.0]`, off | `0.0`, off |
| `snn_final` | `snn` | 64 | 4.0 s (201 frames) | 2, up to 8 bands | 2, up to 10 frames (0.2 s) | `[1.0]`, off | `0.0`, off |
| `cnn_final_3` | `cnn` | 40 | 10.0 s (1001 frames) | 2, up to 8 bands | 2, up to 20 frames (0.2 s) | `[1.0]`, off | `0.0`, off |
| `cnn_base_session_out_2` (baseline) | `cnn` | 64 | 5.0 s (501 frames) | 0 (width 0), off | 0 (width 0), off | absent, so `[1.0]` | absent, so `0.0` |

- All five share `win_ms` 25.0, `n_fft` 512, `f_min` 0.0, `f_max` 8000.0, `log_offset` 1e-06 and `normalization` "utterance".
- `hop_ms` is 10.0 (100 frames per second) everywhere except `snn_final`, which uses 20.0 (50 frames per second).
- Frame counts follow `1 + samples // hop`, so a `threeblock_final` training batch is a float32 tensor of shape `[16, 40, 1001]`.
- `cnn_lif` is the CNN + LIF hybrid and `snn` the fully spiking CNN + LIF ([catalog.py lines 12 to 13](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/catalog.py#L12-L13)).
- `cnn_base_session_out_2` is the baseline that `cnn_final_3` names. The other three name different base hashes. The 3-block base is the record in `threeblock_session_out_2`; the LIF and SNN bases match no record there.

### Questions she may ask about the input

- **Is SpecAugment ever applied to validation or test data?** No. Its only call is in `Step.loss` ([train/loop.py line 185](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L185)), which runs on training batches. `predict` ([lines 218 to 243](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L218-L243)) passes features straight from the normalizer to the model.
- **What does per-utterance normalization remove, and can it leak test data?** Subtracting each band's mean removes the utterance's overall level and average spectral shape, and dividing by each band's SD removes how widely that band varies; the pattern of changes within the utterance remains. So the overall loudness that the comment on [logmel.py line 31](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L31) calls "an arousal cue" is gone. It uses only the utterance's own frames, so nothing passes between utterances ([train/loop.py lines 124 to 135](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L124-L135)).
- **Does the model see the same input in training and evaluation?** Not exactly. Training uses a random `crop_s` crop, normalized over the crop's real frames, then masked. Evaluation uses each whole utterance, normalized over all of it, with no masks.
- **How long is a time mask in seconds?** `time_mask_width` counts frames ([augment/settings.py line 21](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/settings.py#L21)), so its length depends on `hop_ms`. The `snn_final` limit of 10 frames at a 20 ms hop is 0.2 s, the same as 20 frames at 10 ms.
- **Why add 1e-06 before the log, and does the log's base matter?** The offset keeps silence and padding finite, at about -13.8 ([logmel.py line 24](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L24)). Another base would multiply every value by a constant, which the z-score removes (up to the 1e-5 in the variance).
- **How is the filter bank checked?** A test compares `mel_filterbank` with `librosa.filters.mel(htk=False, norm="slaney")`, including `n_fft` 512 with 64 or 40 bands up to 8000 Hz ([tests/test\_torch\_parts.py lines 14 to 23](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_torch_parts.py#L14-L23)). Another compares the whole log-mel with librosa and checks for `1 + n // hop` frames ([lines 25 to 45](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_torch_parts.py#L25-L45)).

## The CNN: models/cnn.py

This file builds the `cnn` model: convolution blocks over the log-mel image, a padding-aware summary over time, and a classifier with one logit per class.

**[Lines 1 to 14](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L1-L14): the module docstring.** It describes each block as \[convolution, normalization, activation\] repeated `convs_per_block` times, then max pooling over frequency and time. After the last block, frequency is averaged, time is summarized by `pooling` over real frames only, and a classifier gives one logit per class. `norm`, `norm_groups`, `norm_subbands` and `activation` were added on 2026-09-26, after Dr. Luitel asked about batch normalization. Settings without them mean batch norm and ReLU, keep their configuration hashes, and build the same network, so a seed gives the same weights.

**[Lines 15 to 21](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L15-L21): imports and accepted names.** `dataclasses` and `Literal` declare the settings, and `register_model` adds the model to the registry. `NORMS` and `ACTIVATIONS` list the names that line 66 accepts. PyTorch is imported only inside the functions that use it, so `src/affect/runs/spec.py` can run `check` at enqueue time without torch (its line 94).

**[Lines 24 to 34](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L24-L34): the settings class, architecture fields.** `CNNSettings` is a frozen (read-only) dataclass with no defaults; values come from the settings file, apart from the fill-ins at lines 46 to 47. `channels` gives one block per entry and that block's output channels; `convs_per_block`, `kernel_size`, `pool_freq` and `pool_time` shape each block. `pooling` picks the summary over time, `classifier_hidden` adds an optional hidden layer, and `dropout` is applied before each of the classifier's linear layers. The validator stores lists as tuples, so `channels` arrives as, for example, `(64, 128, 256)`.

**[Lines 35 to 44](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L35-L44): the settings class, normalization and activation fields.** `norm` names the layer after each convolution: `"batch"` (Ioffe and Szegedy 2015), `"subspectral"` (Chang et al. 2021), `"group"` (Wu and He 2018) or `"none"`. Sub-spectral norm is batch norm computed separately in each sub-band (a run of adjacent frequency rows), and `"none"` drops normalization and gives the convolutions a bias. `norm_groups` and `norm_subbands` hold the group or sub-band count (line 38: one group is layer norm) and are `None` when unused. `activation` follows every normalization and the head's hidden layer; leaky ReLU has slope 0.01, ELU has alpha 1, and GELU is the exact erf form.

**[Lines 46 to 47](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L46-L47): values for settings written before 2026-09-26.** `ADDED_FIELDS` has no type annotation, so it is a class attribute, not a setting. A settings dict with none of the four new keys gets `norm="batch"`, `norm_groups=None`, `norm_subbands=None` and `activation="relu"`. A dict with only some of them is refused. The hash skips these fields while they hold these values, so both forms of one network share a hash (src/affect/settings.py, `canonical_plain`).

**[Lines 50 to 67](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L50-L67): checks of the convolutional front end.** `channels` must be a non-empty list of positive numbers, `convs_per_block` at least 1, `kernel_size` odd, and both pool factors at least 1. Lines 60 to 65 divide `n_mels` by `pool_freq` once per block, rounding down, and refuse a stack that leaves no frequency band. Line 66 refuses names outside `NORMS` and `ACTIVATIONS`, repeating the `Literal` check made when settings are read from a file. The CNN + Transformer and CNN + LIF models call this function too.

**[Lines 68 to 83](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L68-L83): checks that match each norm to its count.** `"group"` needs `norm_groups` of at least 1 that divides every block's channel count, with `norm_subbands` left `None`. `"subspectral"` needs `norm_subbands` of at least 2, `norm_groups` `None`, and at least that many frequency bands entering every block (`n_mels // pool_freq ** block`). `"batch"` and `"none"` need both counts to be `None`.

**[Lines 86 to 95](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L86-L95): checks of the head, and the combined check.** `dropout` must lie in \[0, 1), and `classifier_hidden` must be `None` or at least 1. `check` runs both sets of checks. `check_job` in src/affect/runs/spec.py calls it when a sweep is enqueued (its lines 103 to 107), and the model calls it again when built (line 220).

**[Lines 98 to 106](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L98-L106): masked pooling, the mask and the mean.** `masked_pool` takes `h` as `[batch, channels, frames]` and a boolean `mask` as `[batch, frames]`, True on real frames. These frames are the time steps left after the blocks, so this page calls them steps. Line 102 turns the mask into 1.0 and 0.0, and line 103 counts each utterance's real steps, at least 1.0 so nothing divides by zero. The mean adds up real steps only and divides by that count; `"mean"` returns it, one value per channel.

```latex
n = \max\Big(1,\ \sum_{t} m_t\Big), \qquad \mu_c = \frac{1}{n} \sum_{t} m_t\, h_{c,t}, \qquad m_t = 1 \text{ on real steps},\ 0 \text{ on padding}
```

**[Lines 107 to 112](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L107-L112): max, and mean plus max.** Padded steps are filled with `torch.finfo(h.dtype).min`, the most negative finite number (about -3.4e38 in float32), so they never win the maximum. `"max"` returns each channel's largest value over real steps, and `"meanmax"` concatenates the mean and the max, `2 × channels` values.

```latex
M_c = \max_{t:\ m_t = 1} h_{c,t}
```

**[Lines 113 to 115](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L113-L115): statistics pooling, mean and SD (`"stats"`).** Line 114 squares each real step's deviation from the masked mean, sums them and divides by `count`. That is the biased (population) variance, dividing by n rather than n - 1. Line 115 adds `1e-6` to the variance inside the square root, then concatenates the means and the SDs, `2 × channels` values. The SD adds how much each channel varies over the utterance, which the mean alone does not show.

```latex
\sigma_c = \sqrt{\frac{1}{n} \sum_{t} m_t\, (h_{c,t} - \mu_c)^2 + 10^{-6}}, \qquad \text{output} = [\mu_1, \dots, \mu_C,\ \sigma_1, \dots, \sigma_C]
```

**[Lines 116 to 121](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L116-L121): attention pooling, and the error case.** A linear layer from the last block's channels to 1 (made at line 226) scores each step, and `.float()` makes the scores float32 even under bf16 autocast. Padded steps get a score of minus infinity, so their softmax weight is exactly 0. The output is the weighted average of the steps, one value per channel; any other `kind` raises `ValueError`.

```latex
s_t = \mathbf{w}^{\top} \mathbf{h}_t + b, \qquad \alpha_t = \frac{m_t\, e^{s_t}}{\sum_{\tau} m_{\tau}\, e^{s_{\tau}}}, \qquad \mathbf{e} = \sum_{t} \alpha_t\, \mathbf{h}_t
```

**[Lines 124 to 125](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L124-L125): the width after pooling.** `"meanmax"` and `"stats"` concatenate two vectors, so they double the width; `"mean"`, `"max"` and `"attention"` keep it.

**[Lines 128 to 141](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L128-L141): the activation options.** Each name maps to one PyTorch layer: `nn.ReLU(inplace=True)`, `nn.LeakyReLU(0.01)`, `nn.ELU(alpha=1.0)`, `nn.GELU()` or `nn.SiLU()`. The activation adds a nonlinearity after every convolution; without it, convolution and batch norm (affine at test time) would compose into affine maps, leaving max pooling as the blocks' only nonlinearity. `inplace=True` lets ReLU overwrite its input to save memory, and an unknown name raises `ValueError`. Below, Φ is the standard normal CDF and σ is the logistic sigmoid.

```latex
\begin{aligned}
\mathrm{relu}(x) &= \max(0, x) \\
\mathrm{leaky\_relu}(x) &= \max(0, x) + 0.01 \min(0, x) \\
\mathrm{elu}(x) &= \begin{cases} x & x > 0 \\ 1.0\,(e^{x} - 1) & x \le 0 \end{cases} \\
\mathrm{gelu}(x) &= x\, \Phi(x) = \tfrac{x}{2}\big(1 + \mathrm{erf}(x / \sqrt{2})\big) \\
\mathrm{silu}(x) &= x\, \sigma(x) = \frac{x}{1 + e^{-x}}
\end{aligned}
```

**[Lines 144 to 162](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L144-L162): sub-spectral normalization.** The class is defined inside a function, so torch is imported only when this norm is used. `torch.tensor_split` cuts the frequency axis (dim 2) into `subbands` nearly equal bands; when the split is uneven, the first bands get one extra row. Each band has its own `nn.BatchNorm2d(channels)`, so statistics, γ and β are per band and channel. The bands are then concatenated back, so the output has the input's shape.

**[Lines 165 to 173](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L165-L173): the front end, setup.** `conv_front_end` returns all blocks as one `nn.Sequential`; `n_mels` is accepted but not used here. Its docstring promises that `"batch"` with `"relu"` gives exactly the layers and order of the CNN before 2026-09-26, which tests/test\_torch\_parts.py checks (line 202). The sub-spectral class is built only when needed, and `c_in = 1` because the log-mel image has one channel. The CNN + Transformer and CNN + LIF models build their front ends with this function.

**[Lines 174 to 177](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L174-L177): the convolution.** The outer loop makes one block per entry of `channels`, and the inner loop adds `convs_per_block` convolutions; only the first changes the channel count. Each is `nn.Conv2d` with a k by k kernel (k = `kernel_size`), stride 1 and `padding=kernel_size // 2`, which keeps frequency and time sizes unchanged. Each output channel o sums a weighted k by k patch of every input channel around (f, t), with zeros beyond the edges. The bias term b\_o exists only when `norm` is `"none"`.

```latex
y_{o}(f, t) = b_o + \sum_{i=1}^{C_{\text{in}}} \sum_{u=-p}^{p} \sum_{v=-p}^{p} W_{o,i}(u, v)\; x_i(f + u,\ t + v), \qquad p = \lfloor k/2 \rfloor
```

**[Lines 178 to 183](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L178-L183): the normalization options.** `"batch"` adds `nn.BatchNorm2d(c_out)`, `"subspectral"` adds the class above, `"group"` adds `nn.GroupNorm(norm_groups, c_out)`, and `"none"` adds nothing. Batch norm standardizes each channel with the mean and biased variance over batch, frequency and time, then applies a learned scale γ and shift β. In eval mode it uses running averages from training instead; only `c_out` is passed, so PyTorch's defaults hold (ε = 1e-5, momentum 0.1). Group norm takes its statistics per utterance over each group of channels, so training and evaluation behave the same.

```latex
\text{batch:}\quad y_{n,c,f,t} = \gamma_c\, \frac{x_{n,c,f,t} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c, \qquad \mu_c,\ \sigma_c^2 \text{ taken over all } (n, f, t)
```

```latex
\text{group:}\quad y_{n,c,f,t} = \gamma_c\, \frac{x_{n,c,f,t} - \mu_{n,g(c)}}{\sqrt{\sigma_{n,g(c)}^2 + \epsilon}} + \beta_c, \qquad \mu_{n,g},\ \sigma_{n,g}^2 \text{ over group } g\text{'s channels and all } (f, t)
```

**[Lines 184 to 185](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L184-L185): activation, then the next input width.** The chosen activation follows every normalization, or the convolution itself when `norm` is `"none"`. `c_in = c_out` makes the next convolution read this one's output channels.

**[Lines 186 to 187](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L186-L187): max pooling, then the finished stack.** `nn.MaxPool2d((pool_freq, pool_time))` keeps the largest value in each non-overlapping tile of P\_f = `pool_freq` bands by P\_t = `pool_time` frames. The stride equals the tile and there is no padding, so each size is divided by the tile and rounded down, and leftover rows or columns at the end are dropped (with 2, the last one when the size is odd). Halving both axes, as 2 and 2 do, lets later kernels cover more time and frequency at lower cost. For the 3-block finalist, line 187 returns 12 layers: Conv2d, BatchNorm2d, ReLU and MaxPool2d, three times.

```latex
z_c(f', t') = \max_{0 \le a < P_f,\ 0 \le b < P_t} y_c(P_f f' + a,\ P_t t' + b), \qquad F' = \Big\lfloor \frac{F}{P_f} \Big\rfloor, \quad T' = \Big\lfloor \frac{T}{P_t} \Big\rfloor
```

**[Lines 190 to 200](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L190-L200): the classifier head.** The head opens with `nn.Dropout(dropout)`, which in training zeroes each input with probability p = `dropout` and scales the rest by 1/(1 - p). With `classifier_hidden` set, it adds `nn.Linear(width, classifier_hidden)`, the activation φ and a second `nn.Dropout(dropout)`. The last layer, `nn.Linear(width, n_classes)`, reads the hidden layer when there is one, since line 198 resets `width` to `classifier_hidden`. The head thus maps the pooled embedding **e** to raw logits; there is no softmax here. Dropout does nothing in eval mode, so at test time the head is just the linear layers and φ.

```latex
\begin{aligned}
\text{no hidden layer:}\quad & \text{logits} = W\, \mathrm{drop}(\mathbf{e}) + \mathbf{b} \\
\text{hidden layer:}\quad & \text{logits} = W_2\, \mathrm{drop}\big(\phi(W_1\, \mathrm{drop}(\mathbf{e}) + \mathbf{b}_1)\big) + \mathbf{b}_2
\end{aligned}
```

**[Lines 203 to 210](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L203-L210): real frames left after pooling.** For each block, the real-frame count is divided by `pool_time` and rounded down, as `MaxPool2d` does to the tensor's time axis. The result is clamped to at least 1, so every utterance keeps at least one step. The CNN + Transformer, CNN + LIF and SNN models import this function too.

```latex
\text{keep} = \max\Big(1,\ \Big\lfloor \frac{L}{\text{pool\_time}^{B}} \Big\rfloor\Big), \qquad L = \text{real frames}, \quad B = \text{number of blocks}
```

**[Lines 213 to 230](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L213-L230): the model class and its constructor.** The class is defined inside `_module_class`, so torch is imported only when a model is built. The constructor checks the settings, builds the blocks, and stores the block count, `pool_time` and `pooling`. With `"attention"` it adds `nn.Linear(channels[-1], 1)`, then sizes the head with `pooled_width`. `embedding_dim` and `head_dropout` are kept for the optional gender head (src/affect/train/loop.py, line 153).

**[Lines 232 to 233](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L232-L233): frames after the blocks.** This method calls `frames_after` with the model's block count and `pool_time`.

**[Lines 235 to 242](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L235-L242): the embedding.** `x` arrives as `[batch, n_mels, frames]`, and `unsqueeze(1)` adds one image channel; unless `normalization` is `"none"`, the loop has already set padded frames to 0. The blocks return `[batch, channels, bands, frames']`, and `.float()` gives the float32 that `masked_pool` expects, even under bf16 autocast (src/affect/train/loop.py, line 193). `.mean(dim=2)` averages the remaining frequency bands, and line 241 marks each utterance's first `keep` steps as real. `masked_pool` then returns `[batch, embedding_dim]`.

**[Lines 244 to 251](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L244-L251): classify, forward, and the returned class.** `classify` applies the head to an embedding, and `forward(x, valid_frames)` returns `classify(embed(x, valid_frames))`. `valid_frames` holds each utterance's real frame count before padding, shape `[batch]`. Keeping the two steps apart lets the loop send one embedding to both the emotion head and a gender head (src/affect/train/loop.py, lines 197 to 199). Line 251 hands the class to `build_cnn`.

**[Lines 254 to 256](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L254-L256): registration.** `@register_model("cnn", settings=CNNSettings)` registers `build_cnn` under the name `"cnn"` and checks `CNNSettings` on import: frozen, no defaults, supported types. A `model` section with `"name": "cnn"` is then validated against `CNNSettings` (src/affect/registry.py, `validate_section`). The loop calls `build_cnn(settings, n_mels, n_classes)` through `MODELS.get(...)` (src/affect/train/loop.py, line 351). Line 10 of src/affect/catalog.py imports this module, which runs the decorator.

### How each setting maps to code

Each field of `CNNSettings`, with the lines of this file that use it:

- `channels`: one block per entry (line 174) and that block's output channels (line 176). Its length is `n_blocks` (line 222), and its last entry is the width before pooling (line 225).
- `convs_per_block`: how many convolution, norm and activation layers each block has (line 175).
- `kernel_size`: the square kernel and its padding, `kernel_size // 2` (line 176). Line 56 requires it odd, so the padding keeps the size.
- `pool_freq` and `pool_time`: the max-pooling tile at the end of every block (line 186). `pool_time` also shrinks the real-frame counts (lines 203 to 210, via lines 223 and 233). `pool_freq` drives the band checks (lines 60 to 65 and 78).
- `pooling`: the summary in `masked_pool` (line 242), the attention layer (line 226) and the doubled width (line 227).
- `classifier_hidden`: the optional hidden `Linear`, activation and `Dropout` (lines 195 to 198).
- `dropout`: the probability of every `nn.Dropout` in the head (lines 194 and 197), also kept as `head_dropout` (line 229).
- `norm`: the layer after each convolution (lines 178 to 183) and whether convolutions have a bias (line 177).
- `norm_groups`: the group count of `nn.GroupNorm` (line 183).
- `norm_subbands`: the band count of `SubSpectralNorm` (line 181).
- `activation`: the nonlinearity after each norm (line 184) and in the hidden layer (line 196).
- `n_mels` and `n_classes` are not fields. The loop passes `features.n_mels` and the label set's class count (src/affect/train/loop.py, lines 330 and 351).
- Fixed in the code, not settings: stride 1, zero padding, max pooling in every block, and the frequency average after the last block. Also fixed: the `1e-6` in `"stats"`, leaky ReLU's 0.01, ELU's 1.0, and PyTorch's default ε and momentum.

### Worked example: tensor shapes

**The 3-block finalist** is selection `threeblock_final` (sweep `cnn_3block_confirm_1`, config hash `e0c211fcced3`). Its blocks use `channels` \[64, 128, 256\], `convs_per_block` 1, `kernel_size` 5, `pool_freq` 2 and `pool_time` 2, with `norm` "batch" and `activation` "relu". Its head uses `pooling` "stats", `classifier_hidden` 128 and `dropout` 0.2. Elsewhere it sets `n_mels` 40, `hop_ms` 10.0, `crop_s` 10.0, `batch_size` 16 and `label_set` "four\_merged" (four classes).

The sample rate is 16000 Hz, IEMOCAP's expected format (src/affect/data/iemocap.py, lines 87 and 551). `window_samples` turns `hop_ms` 10.0 into a hop of 160 samples (src/affect/features/logmel.py, line 72), so there are 100 frames per second. The STFT is centred, so n samples give 1 + n // 160 frames (line 111). In training, each utterance becomes `crop_s` × 16000 = 160000 samples: a random crop if longer, zero-padded if shorter (src/affect/train/loop.py, line 364; src/affect/train/data.py, `crop_arrays`).

```latex
\text{hop} = \mathrm{round}\Big(\frac{10.0 \times 16000}{1000}\Big) = 160, \qquad \frac{16000}{160} = 100 \text{ frames per second}, \qquad T = 1 + \Big\lfloor \frac{160000}{160} \Big\rfloor = 1001
```

| step | shape | note |
| --- | --- | --- |
| waveform crops | `[16, 160000]` | `batch_size` 16, 10.0 s at 16000 Hz |
| log-mel `x` | `[16, 40, 1001]` | `n_mels` 40, 1 + 160000 // 160 frames |
| `x.unsqueeze(1)` | `[16, 1, 40, 1001]` | one image channel |
| block 1: conv, BN, ReLU | `[16, 64, 40, 1001]` | `Conv2d(1, 64, 5, padding=2, bias=False)` keeps the size |
| block 1: max pool | `[16, 64, 20, 500]` | 40 // 2 and 1001 // 2: the last frame is dropped |
| block 2: conv, BN, ReLU | `[16, 128, 20, 500]` | `Conv2d(64, 128, 5, padding=2, bias=False)` |
| block 2: max pool | `[16, 128, 10, 250]` | 20 // 2 and 500 // 2 |
| block 3: conv, BN, ReLU | `[16, 256, 10, 250]` | `Conv2d(128, 256, 5, padding=2, bias=False)` |
| block 3: max pool | `[16, 256, 5, 125]` | one step per 8 frames (80 ms) |
| mean over frequency | `[16, 256, 125]` | the 5 remaining bands averaged |
| mask | `[16, 125]` | `keep` = real frames // 8; a full crop keeps all 125 |
| stats pooling | `[16, 512]` | 256 means, then 256 SDs (`embedding_dim` 512) |
| `Dropout(0.2)`, `Linear(512, 128)`, ReLU | `[16, 128]` | the hidden layer |
| `Dropout(0.2)`, `Linear(128, 4)` | `[16, 4]` | logits for neu, hap+exc, ang, sad |

A 3.0 s utterance has 48000 samples, so 1 + 48000 // 160 = 301 real frames. It keeps 301 // 8 = 37 of the 125 steps, and the other 88 are masked. Validation and test feed whole utterances, up to 32 per batch, padded to the longest (src/affect/train/data.py, `eval_batches`). There 1001 becomes the batch's longest frame count, and the same rules apply.

The finalist has 1,092,676 parameters, the count that `efficiency.json` records for its runs (for example artifacts/runs/cnn\_3block\_confirm\_1/e0c211fcced3\_test\_S4\_seed9):

```latex
\underbrace{25\,(1 \cdot 64 + 64 \cdot 128 + 128 \cdot 256)}_{\text{convolutions: } 1{,}025{,}600} + \underbrace{2\,(64 + 128 + 256)}_{\gamma,\ \beta:\ 896} + \underbrace{(512 \cdot 128 + 128) + (128 \cdot 4 + 4)}_{\text{head: } 66{,}180} = 1{,}092{,}676
```

**The 4-block baseline.** Selection `cnn_base_session_out_2` holds it; `base_protocol_utterance_5fold` has the same features, model and training settings, with only `data.split` changed. It differs in `n_mels` 64, `channels` \[32, 64, 128, 256\], `kernel_size` 3 (padding 1), `pooling` "mean", `classifier_hidden` null and `dropout` 0.3. It trains on `crop_s` 5.0 with `batch_size` 32. Its model section predates `norm` and `activation`, so `ADDED_FIELDS` supplies batch norm and ReLU. A 5.0 s crop is 80000 samples, so `x` is `[32, 64, 501]`. The four blocks give `[32, 32, 32, 250]`, `[32, 64, 16, 125]`, `[32, 128, 8, 62]` and `[32, 256, 4, 31]`. Each final step stands for 16 frames (160 ms). Averaging frequency gives `[32, 256, 31]`, mean pooling gives `[32, 256]`, and `Dropout(0.3)` with `Linear(256, 4)` gives `[32, 4]`. It has 389,348 parameters.

**cnn\_final\_3.** It keeps the baseline's four blocks \[32, 64, 128, 256\], `pooling` "mean", no hidden layer and `dropout` 0.3. It shares `n_mels` 40, `kernel_size` 5, `crop_s` 10.0 and `batch_size` 16 with the 3-block finalist, and spells out `norm` "batch" and `activation` "relu". From `[16, 40, 1001]` the blocks give `[16, 32, 20, 500]`, `[16, 64, 10, 250]`, `[16, 128, 5, 125]` and `[16, 256, 2, 62]`. In block 4, rounding down drops a band (5 // 2 = 2) and a step (125 // 2 = 62). Then come `[16, 256, 62]`, `[16, 256]` and `[16, 4]`. It has 1,077,988 parameters; both counts match `n_parameters` in their runs' `efficiency.json` files.

### Questions she may ask about this file

- **Is padding ignored everywhere?** No. In the model, only the summary over time ignores it; before the model, the normalizer leaves padded frames out of its statistics and sets them to 0 (src/affect/train/loop.py, lines 67 and 82). The blocks run on the whole padded tensor: convolutions near a short utterance's end see padding, and batch norm's training statistics include it. With `crop_s` 10.0, every training utterance shorter than 10 s is padded to 160000 samples (1001 frames). `masked_pool` then uses only the first `keep` steps (lines 240 to 242).
- **What exactly does SD pooling (`"stats"`) compute?** For each channel, the mean and the SD over the utterance's real steps, concatenated: 512 values for the finalist. The variance divides by the number of real steps, so it is biased (not n - 1). Adding `1e-6` inside the square root keeps the SD and its gradient finite when the variance is 0, for example with one real step.
- **What does batch norm do at validation and test time?** `predict` calls `model.eval()` (src/affect/train/loop.py, line 224), so `BatchNorm2d` uses the running mean and variance from training. Batch norm then treats each utterance the same way, whatever else is in its batch. In training (`model.train()`, line 384) it uses the current batch's statistics and updates the running averages.
- **Why do the convolutions have no bias?** Batch norm and sub-spectral norm subtract each channel's mean, which removes any constant bias, and their β adds a learned shift back. Line 177 adds a bias only for `"none"`. Group norm also gets none, though its shared group mean would cancel a per-channel bias only partly.
- **Where is the softmax?** Not in this file: `forward` returns logits. The loop applies `log_softmax` inside the loss (src/affect/train/loop.py, line 141) and `softmax` for predicted probabilities (line 244).
- **How much of the spectrogram does one pooled step see?** Each convolution widens the view by (k - 1) times the current spacing; each 2 by 2 pooling adds one spacing, then doubles it. For the 3-block finalist, one cell of the last block's output sees 36 frames (about 0.36 s) by 36 of the 40 mel bands, with steps 8 frames apart. The 4-block baseline's cells see 46 frames by 46 of its 64 bands, and cnn\_final\_3's see 76 frames and all 40 bands. The average over frequency then combines cells that together cover every band, so each step that the pooling summarizes depends on all the bands but only on that span of frames.

## The LIF neuron: models/lif.py

This section reads `lif.py` in line order, then covers how the models feed it, the numpy reference that checks it, and the finalists' settings.

### lif.py, line by line

**[Lines 1 to 28](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L1-L28): the module docstring.** It states the rules the code implements, for membrane potential U, input current I, spike S, leak β (`beta`) and threshold θ (`threshold`). Each step keeps the fraction β of the old potential, adds the new current, and fires a spike when U reaches θ. Every neuron argument is an explicit model setting; nothing relies on a library default. The state stays float32 even under bf16 mixed precision (both finalists train in bf16), and time runs as a plain Python loop.

```latex
\begin{aligned}
\text{subtract reset:}\quad U[t] &= \beta\,U[t-1] + I[t] - \theta\,S[t-1] \\
\text{zero reset:}\quad U[t] &= \beta\,(1 - S[t-1])\,U[t-1] + I[t] \\
\text{spike:}\quad S[t] &= \begin{cases} 1 & \text{if } U[t] \ge \theta \\ 0 & \text{otherwise} \end{cases}
\end{aligned}
```

**[Lines 29 to 34](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L29-L34): imports and option lists.** `math` supplies π for the atan surrogate. `RESETS` (`"subtract"`, `"zero"`) and `SURROGATES` (`"atan"`, `"fast_sigmoid"`, `"sigmoid"`, `"boxcar"`) come from the numpy reference, so both files accept the same options. `LEARN_BETA` lists the leak options (`"none"`, `"layer"`, `"neuron"`) and `LIF_NORMS` the current normalisations (`"none"`, `"batch"`, `"layer"`). Torch is imported only inside functions, so this module loads without it.

**[Lines 37 to 60](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L37-L60): settings validation (`check_neuron`).** It raises `ValueError` for any neuron setting outside its allowed range. `beta`, `readout_beta`, `lif_dropout` and `dropout` must lie in \[0, 1), and `threshold` and `surrogate_slope` must be above 0. `rate_weight` must be at least 0, `rate_target` in \[0, 1\], each `lif_hidden` size at least 1, and each option on its list. `cnn_lif.check` and `snn.check` call it, and `check_job` runs those without torch when a job is queued ([runs/spec.py lines 93 to 107](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py#L93-L107)).

**[Lines 63 to 74](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L63-L74): the spike, forward pass.** `Spike` is a custom `torch.autograd.Function`, an operation whose forward and backward passes are written by hand. Its forward receives x = U - θ, saves x for the backward pass, and returns 1 where x ≥ 0, else 0. This is the Heaviside step H, so a potential exactly at θ fires.

```latex
S[t] = H\bigl(U[t] - \theta\bigr), \qquad H(x) = \begin{cases} 1 & x \ge 0 \\ 0 & x < 0 \end{cases}
```

**[Lines 76 to 93](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L76-L93): the surrogate gradient, backward pass.** The true derivative of H is 0 everywhere except at x = 0, so exact backpropagation would pass no signal back through any spike. Instead, the backward pass multiplies the incoming gradient by a bump sg(x), with a = `surrogate_slope`; the forward spike stays a hard 0 or 1. Both finalists use `"boxcar"` with slope 1.0: sg is 0.5 while |U - θ| < 0.5, and 0 elsewhere. The two `None` returns mean `kind` and `slope` receive no gradient.

```latex
\mathrm{sg}(x) =
\begin{cases}
\dfrac{a/2}{1 + \left(\frac{\pi}{2}\,a\,x\right)^{2}} & \text{atan} \\[2ex]
\dfrac{1}{\left(1 + a\,|x|\right)^{2}} & \text{fast\_sigmoid} \\[2ex]
a\,\sigma(a x)\,\bigl(1 - \sigma(a x)\bigr) & \text{sigmoid} \\[1ex]
0.5\,a \ \text{ if } |x| < 0.5/a, \ \text{else } 0 & \text{boxcar}
\end{cases}
```

**[Lines 96 to 103](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L96-L103): the `spike` helper.** It builds the `Spike` class on first use, keeps it in a module global, and applies it. The neuron loop calls it at line 221 with x = U - θ.

**[Lines 106 to 110](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L106-L110): `step_mask`.** It returns a `[batch, steps]` table of true or false, true on each sequence's real steps. Padding always follows an utterance's own steps, so step t is real when t < `valid_steps`.

**[Lines 113 to 131](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L113-L131): `pool_time`, averaging groups of steps.** With factor k, it zero-pads the sequence to a multiple of k, then averages each group of k consecutive steps over its real steps only. With mask m (1 on real steps), a group without real steps gives 0 and is marked as padding. Factor 1 returns the input unchanged. Despite the name, this implements the `time_pool` setting (4 in the hybrid finalist), not the front end's `pool_time` max-pool.

```latex
\bar h[g] = \frac{\sum_{j=0}^{k-1} m[gk+j]\; h[gk+j]}{\max\!\left(1,\ \sum_{j=0}^{k-1} m[gk+j]\right)}
```

**[Lines 134 to 148](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L134-L148): the readout, a leaky integrator with no threshold.** Each last-layer neuron's spikes feed Y\[t\] = β\_r Y\[t-1\] + S\[t\], starting from Y = 0, where β\_r is `readout_beta`. The embedding is (1 - β\_r) times the mean of Y over the T real steps; with β\_r = 0 it is spikes per step. The scale makes a steady rate r read as about r on long sequences; summed out, a spike at step k weighs 1 - β\_r^(T-k). So late spikes count less: over 25 steps at the spiking finalist's 0.97, weights fall from 0.533 (first step) to 0.03 (last).

```latex
\begin{aligned}
Y[t] &= \beta_r\,Y[t-1] + S[t], \qquad Y[-1] = 0 \\
e &= \frac{1-\beta_r}{T}\sum_{t=0}^{T-1} Y[t] \;=\; \frac{1}{T}\sum_{k=0}^{T-1}\bigl(1-\beta_r^{\,T-k}\bigr)\,S[k]
\end{aligned}
```

**[Lines 151 to 158](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L151-L158): `time_dropout`.** In training, it draws one keep-or-drop mask per sequence and unit, and applies the same mask at every step. A dropped unit therefore stays silent for the whole utterance, and kept values are scaled by 1/(1 - p). Outside training, or with p = 0, it returns the input unchanged.

**[Lines 161 to 174](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L161-L174): `Recorder`.** It is a small accumulator for efficiency.json: while `on` is true, `add` sums a number under a name. `reset` clears the sums and the utterance count. Training switches it on only for the final validation pass, with the best weights loaded ([train/loop.py lines 451 to 461](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L451-L461)).

**[Lines 177 to 201](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L177-L201): `Neuron`, a layer of LIF neurons.** It maps a current `[batch, steps, ...]` to spikes of the same shape, in float32. The threshold, reset rule, surrogate and penalty settings are stored as plain values, so the threshold is never learned. For `learn_beta` `"none"`, β is a buffer (saved, not trained); `"layer"` learns one β, and `"neuron"` one per neuron, all starting at `beta`. The `real` mask changes nothing in the dynamics, only what the penalty and the recorder count.

**[Lines 203 to 204](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L203-L204): the leak clamp.** `beta()` clamps the stored leak to \[0, 1\] each time it is used. A learned leak may drift outside that range, but the dynamics never see such a value; at 1 the neuron would keep its whole potential.

**[Lines 206 to 211](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L206-L211): number type and initial state.** The current is cast to float32 (float64 passes through, for the tests). U and S start as zeros at every forward call, so nothing carries over between batches or utterances.

```latex
U[-1] = 0, \qquad S[-1] = 0
```

**[Lines 212 to 222](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L212-L222): the time loop: integrate, reset, fire.** Each step takes the current I\[t\]; with `recurrent` true it adds R S\[t-1\], a learned linear map of the layer's previous spikes. `r` is the previous step's spike, detached from the gradient when `detach_reset` is true. Line 218 applies the subtract rule, line 220 the zero rule, and line 221 fires, so a reset shows one step after its spike. The zero rule drops the carried potential but still adds the step's input, so U\[t\] equals that input right after a spike.

```latex
I_{\text{in}}[t] = I[t] + R\,S[t-1] \qquad \text{(recurrent layers only)}
```

**[Lines 223 to 230](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L223-L230): stacking and the activity penalty.** The per-step spikes are stacked into `[batch, steps, ...]`. In training with `rate_weight` above 0, each unit's rate is its spike count over the batch's real steps, divided by the number of real steps. A convolution layer also averages over its frequency rows, giving one rate per channel. The layer's penalty is the mean over units of (rate - `rate_target`)²; outside training it stays `None`.

```latex
r_j = \frac{\sum_{b,t} m_{b,t}\,S_{b,t,j}}{\sum_{b,t} m_{b,t}}, \qquad
P_\ell = \frac{1}{N_\ell}\sum_{j=1}^{N_\ell}\bigl(r_j - r^{*}\bigr)^{2}, \qquad r^{*} = \text{rate\_target}
```

**[Lines 231 to 234](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L231-L234): recording and return.** With the recorder on, the layer adds its spike count and its number of real neuron-steps, both over real positions only. It then returns the spikes.

**[Lines 236 to 246](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L236-L246): `StepBatchNorm`.** It applies batch norm to a `[batch, steps, width]` current, over real steps only. It picks the real steps with the mask, normalises them with `BatchNorm1d`, and leaves padded steps at 0. The spiking finalist uses it (`lif_norm` `"batch"`).

**[Lines 248 to 261](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L248-L261): `SpikingLinear`, one spiking layer.** It holds a linear map `fc`, an optional normaliser, an optional bias-free recurrent map R, `n_out` LIF neurons, and a dropout rate. `fc` has a bias only with `lif_norm` `"none"`; both normalisers have their own learned shift. `real_input` records whether the input is real numbers or spikes, for the operation counts.

**[Lines 263 to 279](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L263-L279): `SpikingLinear.forward`.** It counts operations, computes the current, normalises it, zeroes padded steps, runs the neurons with R, and applies `time_dropout`. A real-valued input counts real steps × `n_in` × `n_out` multiply-accumulates; a spike input counts input spikes × `n_out` accumulates. With R, lines 277 to 278 also add the layer's own spikes on real steps × `n_out` accumulates. In the formula below, b exists only with `lif_norm` `"none"`, where Norm does nothing.

```latex
I[t] = m[t]\cdot\mathrm{Norm}\bigl(W\,x[t] + b\bigr)
```

**[Lines 281 to 291](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L281-L291): building the classes once.** `_module_classes` returns `Neuron`, `StepBatchNorm` and `SpikingLinear`, and `classes()` builds them on the first call only. Like `spike`, this defers importing torch until a model is built.

**[Lines 294 to 305](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L294-L305): `penalty_of`, the term added to the loss.** It returns `None` when the weight is 0 or no layer stored a penalty, as happens outside training. Otherwise it averages the stored penalties over all `Neuron` layers (the set N below) and multiplies by `rate_weight` (λ). The training step adds the result to the classification loss ([train/loop.py lines 200 to 204](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L200-L204)).

```latex
\mathcal{L} = \mathcal{L}_{\text{class}} + \lambda\,\frac{1}{|\mathcal{N}|}\sum_{\ell \in \mathcal{N}} P_\ell
```

**[Lines 308 to 323](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L308-L323): `report`, the spike section of efficiency.json.** For each neuron layer it gives the firing rate, spikes divided by real neuron-steps, and lists layers with rate 0 as silent. It also gives operations per sequence and an energy estimate at 4.6 pJ per multiply-accumulate and 0.9 pJ per accumulate (Horowitz 2014, 45 nm). Its own note says memory access is ignored.

```latex
E\ [\mu\text{J per sequence}] = \frac{4.6\,N_{\text{MAC}} + 0.9\,N_{\text{AC}}}{10^{6}}
```

### From CNN frames to steps, and from spikes to class scores

**[cnn\_lif.py lines 90 to 100](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L90-L100): the hybrid's spiking stack.** The first spiking layer's input width is the front end's last channel count, times the remaining bands only for `"flatten"`. One `SpikingLinear` is built per `lif_hidden` entry, and only the first has `real_input` true. The head is dropout, then one linear layer from the last width to the classes.

**[cnn\_lif.py lines 117 to 133](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L117-L133): from CNN frames to LIF steps.** The ordinary, non-spiking CNN maps the log-mel image to `[batch, channels, bands, steps]`; each block max-pools time by `pool_time`. `"mean"` then averages over frequency, leaving one real-valued vector per step, and `lif.pool_time` (line 130) averages groups of `time_pool` steps. In the hybrid finalist, that gives one 256-value vector per 64 log-mel frames. These vectors are the first spiking layer's input x\[t\], which its weights and normaliser turn into current; the audio is never turned into spikes before it.

**[cnn\_lif.py lines 135 to 139](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L135-L139): class scores.** `classify` sends the embedding e through the head: dropout, then a linear layer with one output per class. Both finalists use the four-class label set, so the head gives 4 logits.

```latex
z = W_{\text{out}}\,\mathrm{Dropout}(e) + b_{\text{out}}
```

**[snn.py lines 160 to 177](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L160-L177): steps in the fully spiking model.** With `encoding` `"frames"`, every log-mel frame is one time step. The first convolution reads normalised log-mel values, so its neurons get real-valued current, and later layers read spikes. Each block max-pools the spikes, so a spike survives if any in its window fired. The last block's spikes, channels times rows, form one vector per remaining step for the spiking linear layers.

### The numpy reference: models/lif\_reference.py

**[lif\_reference.py lines 1 to 32](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif_reference.py#L1-L32): what it is.** It is a plain numpy version of the neuron, its gradient, the four surrogates and the readout, so its own tests run without torch. The torch layers must match it on fixed inputs, forward and backward. It follows Eshraghian et al. 2023 and snnTorch's Leaky neuron: the reset at step t uses the spike of step t - 1. It notes that snnTorch always detaches the reset; here that is the `detach_reset` setting.

**[lif\_reference.py lines 41 to 75](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif_reference.py#L41-L75): surrogates and forward pass.** `surrogate_derivative` repeats the four formulas. `lif_forward` runs the same loop as `Neuron.forward` and returns spikes and potentials. Its `fire` argument can replace the hard step with a smooth function, which the finite-difference tests need.

**[lif\_reference.py lines 78 to 114](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif_reference.py#L78-L114): backpropagation through time, by hand.** `lif_backward` walks from the last step back and returns dL/dI and dL/dβ without autograd. Below, δX means ∂L/∂X, g\[t\] is the gradient arriving from outside the layer, and sg\[t\] = sg(U\[t\] - θ). The θ and βU\[t\] terms carry the gradient through the reset; `detach_reset` sets them to 0.

```latex
\begin{aligned}
\text{subtract:}\quad \delta S[t] &= g[t] - \theta\,\delta U[t+1] \\
\delta U[t] &= \delta S[t]\,\mathrm{sg}[t] + \beta\,\delta U[t+1] \\[1ex]
\text{zero:}\quad \delta S[t] &= g[t] - \beta\,U[t]\,\delta U[t+1] \\
\delta U[t] &= \delta S[t]\,\mathrm{sg}[t] + \beta\,(1 - S[t])\,\delta U[t+1] \\[1ex]
\text{both:}\quad \delta I[t] &= \delta U[t]
\end{aligned}
```

**[lif\_reference.py lines 117 to 125](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif_reference.py#L117-L125): `readout_embedding`.** It computes the same readout as `lif.readout`, for one sequence and its first `valid_steps` steps.

### How the tests compare torch with the reference

**[test\_lif\_reference.py lines 33 to 42](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_lif_reference.py#L33-L42): hand-worked steps.** The input is a constant 0.6, with β = 0.5 and θ = 1.0. The subtract rule must give U = 0.6, 0.9, 1.05, 0.125, 0.6625, firing once at step 2. The zero rule must give 0.6, 0.9, 1.05, 0.6, 0.9. A third case checks that a potential exactly at θ fires.

**[test\_lif\_reference.py lines 50 to 63](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_lif_reference.py#L50-L63): the surrogates.** Each must take a known value at x = 0: atan (slope 2.0) 1.0, fast\_sigmoid (25.0) 1.0, sigmoid (25.0) 6.25, boxcar (1.0) 0.5. Each must be symmetric and never negative. The boxcar at slope 1.0 must pass 0.5 at x = 0.49 and 0 at 0.51. Its area must be 0.5 (at slope 2.0), while sigmoid's must be 1.0 (at slope 4.0).

**[test\_lif\_reference.py lines 65 to 103](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_lif_reference.py#L65-L103): the hand-written gradient against finite differences.** Finite differences of a hard step are useless, so the test swaps in a smooth spike, sigmoid(3x). Its derivative is exactly the `"sigmoid"` surrogate at slope 3. `lif_backward` must then match central finite differences within 1e-7, for both resets and β of 0.0, 0.7 and 0.93. With `detach_reset`, it must match a forward pass whose reset spikes are frozen, and differ from the full gradient.

**[test\_lif\_reference.py lines 105 to 113](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_lif_reference.py#L105-L113): the readout.** With β\_r = 0 it must give spikes per step, counting real steps only. With β\_r = 0.5, the spike train 1, 1, 0, 1 must read 0.578125. A steady rate of 1 over 400 steps at β\_r = 0.9 must read 1.0 within 0.03.

**[test\_lif\_torch.py lines 26 to 64](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_lif_torch.py#L26-L64): torch forward against the reference.** Random currents go through both `Neuron` and `lif_forward`, and the spikes must be exactly equal. This covers both resets, β of 0.0, 0.5 and 0.875, float32 input, and the recurrent map. These tests need torch, so they run on the node.

**[test\_lif\_torch.py lines 66 to 88](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_lif_torch.py#L66-L88): torch gradients against the reference.** For all four surrogates, both resets, and `detach_reset` on and off, the test backpropagates the sum of g·S through `Neuron`. The input gradient must match `lif_backward` within 1e-9, and the learned leak's gradient within 1e-5 times max(1, |dL/dβ|).

**[test\_lif\_torch.py lines 90 to 124](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_lif_torch.py#L90-L124): clamp and penalty.** A learned leak set to 1.7 or -0.2 is used as 1.0 or 0.0, and a fixed leak adds no parameters. The penalty must equal the mean squared rate gap with padded steps left out, and be `None` in eval mode.

**[test\_lif\_torch.py lines 129 to 236](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_lif_torch.py#L129-L236): the other parts.** `pool_time`, `readout` (against `readout_embedding`), `StepBatchNorm`, padding, `time_dropout` and the operation counts each have a test. For example, changing padded inputs must leave every real step's spikes and the readout unchanged (lines 180 to 199).

### The two finalists' LIF settings

Values come from the `settings` field of `artifacts/selections/lif_final/selection.json` and `artifacts/selections/snn_final/selection.json`: the `model` section, plus `hop_ms` from `features` and `crop_s` from `train`. The last row is arithmetic on `crop_s` and `hop_ms` at 16 kHz. Frames are 1 + samples // hop, and each max-pool halves the steps, rounding down.

| Setting | lif\_final (`cnn_lif`, hybrid) | snn\_final (`snn`, fully spiking) |
| --- | --- | --- |
| Leak: `beta`, `learn_beta` | 0.8, `"none"` (fixed) | 0.8, `"layer"` (one learned per layer) |
| `threshold` | 0.5 | 1.0 |
| `reset`, `detach_reset` | `"zero"`, false | `"subtract"`, false |
| `surrogate`, `surrogate_slope` | `"boxcar"`, 1.0 | `"boxcar"`, 1.0 |
| `rate_target`, `rate_weight` | 0.2, 1.0 | 0.2, 1.0 |
| `readout_beta` | 0.5 | 0.97 |
| Spiking layers | one, `lif_hidden` \[256\] (input width 256), after a non-spiking CNN with `channels` \[32, 64, 128, 256\] | six spiking convolutions (`channels` \[16, 32, 64\], `convs_per_block` 2), then `lif_hidden` \[128, 64\] (input width 512) |
| `recurrent`, `lif_norm` | false, `"layer"` | true, `"batch"` |
| `lif_dropout`, `dropout` | 0.3, 0.3 | 0.0, 0.3 |
| Time axis | `hop_ms` 10.0; `pool_time` 2 in each of 4 blocks; `freq_reduce` `"mean"`; `time_pool` 4 | `hop_ms` 20.0; `encoding` `"frames"`; `pool_time` 2 in each of 3 blocks |
| Steps per full training crop | `crop_s` 5.0: 501 frames, 31 front-end steps, 8 LIF steps | `crop_s` 4.0: 201 frames; convolutions at 201, 100 and 50 steps; linear layers at 25 |

### Worked example: one hybrid neuron, three steps

The inputs below are made up for illustration only; they do not come from the trained model. The neuron uses lif\_final's β = 0.8, θ = 0.5, zero reset, and boxcar surrogate at slope 1.0.

| t | I\[t\] (illustrative) | Carried: 0.8 × (1 - S\[t-1\]) × U\[t-1\] | U\[t\] | U\[t\] ≥ 0.5? | S\[t\] | sg(U\[t\] - 0.5) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.3 | 0.8 × 1 × 0 = 0 | 0.3 | no | 0 | 0.5 |
| 1 | 0.4 | 0.8 × 1 × 0.3 = 0.24 | 0.64 | yes | 1 | 0.5 |
| 2 | 0.3 | 0.8 × 0 × 0.64 = 0 | 0.3 | no | 0 | 0.5 |

Neither input alone reaches 0.5, but the leaky sum does at step 1. At step 2 the zero rule drops the old 0.64; the subtract rule would give 0.8 × 0.64 + 0.3 - 0.5 = 0.312 instead. The surrogate is 0.5 at all three steps because U stays strictly between 0 and 1. A potential of 1.0 or more would still spike, but that spike would pass no gradient back to U.

### Questions she may ask about the LIF neuron

- **Why is a surrogate gradient needed, and what does it pass back?** The spike is a step, so its derivative is 0 almost everywhere and no error would reach the weights through it. `Spike.backward` (lines 76 to 91) uses sg(U - θ) instead: for both finalists, 0.5 when |U - θ| < 0.5, else 0. So a spike passes gradient only when U lies strictly between 0 and 1 in the hybrid, or 0.5 and 1.5 in the spiking finalist.
- **When does the reset happen, and how do the two rules differ?** Step t resets with the previous spike S\[t-1\] (lines 216 to 220), as in snnTorch's Leaky neuron. After a spike, subtract carries βU\[t-1\] - θ (possibly negative), while zero carries nothing; both still add I\[t\]. The hybrid uses zero and the spiking finalist subtract, both with `detach_reset` false, so gradient also flows through the reset.
- **Are the leak and the threshold learned?** The threshold never is: it is a plain number (line 190). The hybrid keeps β fixed at 0.8; the spiking finalist learns one β per layer from 0.8, clamped to \[0, 1\] (line 204). The other learned parts of a spiking layer are its input weights, the normaliser's scale and shift (or a bias, when there is no normaliser), and R when `recurrent` is true.
- **How many time steps do the neurons see?** The hybrid sees 8 per full training crop: 5.0 s gives 501 frames, 31 steps after four max-pools, then 8 after `time_pool` 4. Each LIF step stands for 64 frames, 0.64 s at the 10.0 ms hop. The spiking finalist's convolutions see 201, 100 and 50 steps per 4.0 s crop, and its linear layers 25.
- **What does the activity penalty add to the loss?** In training only, each unit's rate over the batch's real steps is compared with `rate_target` 0.2 (lines 226 to 230). Squared gaps are averaged per layer, then over neuron layers (1 in the hybrid, 8 in the spiking finalist), then multiplied by `rate_weight` 1.0. It is two-sided: it pushes silent units up and busy units down.
- **How is the torch code checked?** It is compared with `lif_reference.py`, which needs only numpy. Spikes must match the reference exactly for both resets, and input gradients within 1e-9 for every surrogate and reset. The reference's own hand-written gradient must match finite differences within 1e-7.

## The CNN + LIF hybrid: models/cnn\_lif.py

The hybrid runs the CNN's real-valued convolution blocks, turns their output into a short sequence, and sends it through LIF neurons to a non-spiking readout.

**[Lines 1 to 18](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L1-L18): the module docstring.** It lists the model in order. The CNN front end gives one vector per remaining time step, and consecutive steps can be averaged in groups of `time_pool`. One to three spiking layers follow, then a leaky-integrator readout that never spikes. Lines 14 to 17 add the activity penalty and the counts written to `efficiency.json`.

**[Lines 19 to 25](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L19-L25): imports.** Line 22 takes three helpers from [models/cnn.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py): the front-end checks, the front-end builder and the frame counter. Line 23 takes `bands_after` from models/cnn\_transformer.py, and line 24 takes `check_neuron` from [lif.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py), which has its own section. None of these modules imports PyTorch at load time, so the settings can be checked on a machine without it.

**[Lines 28 to 39](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L28-L39): front-end settings.** `CNNLIFSettings` is a frozen dataclass with no defaults, so every value must be written in the job's settings. These nine fields carry the CNN's front-end names, which `conv_front_end` and `check_front_end` read. `norm` and `activation` apply to the convolution blocks only.

**[Lines 40 to 42](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L40-L42): from feature maps to time steps.** `freq_reduce` decides how the frequency axis goes away: `"mean"` averages the remaining bands, and `"flatten"` keeps every band's channels side by side. `time_pool` averages consecutive steps in groups of that size before the spiking layers; 1 means no averaging.

**[Lines 43 to 54](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L43-L54): spiking-layer settings.** `lif_hidden` gives the width of each spiking layer, and `recurrent` also feeds each layer its own spikes from the previous step. `lif_norm` normalises each layer's input current, and `lif_dropout` drops neurons with one mask per sequence. The last seven fields set the neuron itself (leak, learned leak, threshold, reset, surrogate gradient); the section on lif.py explains them.

**[Lines 55 to 59](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L55-L59): readout and loss settings.** `readout_beta` is the leak of the readout's integrator, in `[0, 1)`. `rate_weight` scales the activity penalty in the loss (0 turns it off), and `rate_target` is the firing rate it pulls each neuron towards. `dropout` acts just before the output layer.

**[Lines 62 to 68](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L62-L68): settings validation.** `check_front_end` ([cnn.py lines 50 to 83](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L50-L83)) checks the front end, for example an odd kernel and at least one frequency band left after pooling. `check_neuron` ([lif.py lines 37 to 60](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L37-L60)) checks the neuron and readout values, for example `beta` and `readout_beta` in `[0, 1)`. Lines 65 to 68 require a `time_pool` of at least 1 and 1 to 3 spiking layers. [runs/spec.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py#L93-L109) runs this check before any job is queued, after [settings.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/settings.py#L217-L242) has rejected missing fields, wrong types and values outside each allowed list.

**[Lines 71 to 77](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L71-L77): PyTorch only when the network is built.** `_module_class` imports PyTorch inside a function, so the settings and checks above load without it. Line 77 fetches the spiking classes from lif.py; this file uses only `SpikingLinear`, and `_` discards the batch-norm helper.

**[Lines 79 to 84](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L79-L84): the CNN front end, reused.** The constructor checks the settings again (line 82) and creates a `Recorder`, which sums spikes and operations only while switched on. Line 84 builds the front end with `conv_front_end` ([cnn.py lines 165 to 187](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L165-L187)), the same function the CNN uses. Each block is `convs_per_block` rounds of convolution, normalisation and activation, then one max pooling over frequency and time.

**[Lines 85 to 89](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L85-L89): settings kept for later.** These lines store the block count, pooling factors, `n_mels`, channels, kernel size, convolutions per block, `freq_reduce` and `time_pool`. `embed`, `frames_after_blocks` and `_front_macs` read them below.

**[Lines 90 to 95](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L90-L95): the LIF layers.** `width` is the length of each step's vector: the last block's channels, times the remaining bands when `freq_reduce` is `"flatten"`. The loop builds one `SpikingLinear` per entry of `lif_hidden`, named `lif1`, `lif2` and so on, each feeding the next. Only the first gets `real_input=True`, because it receives real numbers rather than spikes. That flag changes only the operation count: multiply-accumulates for real input, accumulates for spikes.

**[Lines 96 to 100](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L96-L100): readout settings and the output layer.** The model keeps `readout_beta` and `rate_weight` for later. `embedding_dim` and `head_dropout` let the training loop attach an optional gender head ([train/loop.py line 153](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L153)). `head` is dropout followed by one linear layer from `width` to the number of classes, with no hidden layer.

**[Lines 102 to 103](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L102-L103): real steps after the front end.** `frames_after` ([cnn.py lines 203 to 210](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L203-L210)) floor-divides each utterance's real frame count by `pool_time` once per block, keeping at least 1. The result tells the model which output steps hold audio and which are padding.

**[Lines 105 to 115](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L105-L115): counting the front end's work.** This runs only while the recorder is on, summing the convolutions' multiply-accumulates over each utterance's own frames. Each convolution adds real frames × bands × `c_out` × `c_in` × `kernel_size` squared. Frames and bands then shrink as each pooling shrinks them, and the total goes into `efficiency.json`.

**[Lines 117 to 122](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L117-L122): running the front end.** `embed` takes the normalised log-mel batch `x`, shaped `[batch, n_mels, frames]`, and each utterance's real frame count. While recording, lines 119 to 121 count utterances and front-end operations. Line 122 adds a channel axis of size 1, so the log-mel is a one-channel image, and runs the blocks: `[batch, channels, bands, steps]`.

**[Lines 123 to 127](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L123-L127): frequency pooling or flattening.** With `"mean"`, line 124 averages over the remaining frequency bands and moves time to the second axis: `[batch, steps, channels]`. With `"flatten"`, lines 126 and 127 move time second and join all bands' channels: `[batch, steps, channels × bands]`. Either way each remaining time position becomes one vector, as in the CNN + Transformer ([cnn\_transformer.py lines 124 to 128](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_transformer.py#L124-L128)). `.float()` turns bf16 values from mixed-precision training into 32-bit floats.

**[Lines 128 to 130](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L128-L130): masking padding, then averaging steps.** Line 129 builds a `[batch, steps]` mask m that is True (1) on each utterance's real steps. Line 130 calls `lif.pool_time` ([lif.py lines 113 to 131](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L113-L131)), which averages each group of k = `time_pool` consecutive steps over its real steps only. A group with any real step stays real, so the mask shrinks with the sequence. The finalist's `time_pool` is 4, so this averages every 4 steps.

```latex
\bar{x}_{g} = \frac{\sum_{t=gk}^{gk+k-1} m_{t}\, x_{t}}{\max\left(1,\ \sum_{t=gk}^{gk+k-1} m_{t}\right)}
```

**[Lines 131 to 132](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L131-L132): the LIF layers run.** Each `SpikingLinear` ([lif.py lines 248 to 279](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L248-L279)) applies weights, optional normalisation, zero current on padding, LIF neurons, and dropout with one mask per sequence. An LIF neuron keeps a leaky running sum of its input and emits a 1 when the sum reaches `threshold`, then resets. The exact update rule and its gradient are in the section on lif.py. The output is a spike train, `[batch, steps, layer width]`, which feeds the next layer.

**[Line 133](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L133): the leaky-integrator readout.** `lif.readout` ([lif.py lines 134 to 148](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L134-L148)) feeds the last layer's spikes S into a leaky integrator Y that has no threshold, so nothing spikes. It averages Y over the T real steps and scales by `1 - readout_beta`, so on long sequences a steady rate r gives about r ([lif\_reference.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif_reference.py#L26-L31)). Unrolling the sum, each step's spikes enter with weight `(1 - readout_beta^(T - t)) / T`. For the finalist (`readout_beta` 0.5, T = 8 on a 5.0 s crop), that is 0.996/8 at the first step and 0.5/8 at the last.

```latex
Y_{t} = \beta_{\mathrm{out}}\, Y_{t-1} + S_{t}, \qquad Y_{-1} = 0, \qquad \beta_{\mathrm{out}} = \mathrm{readout\_beta}
```

```latex
z = \frac{1 - \beta_{\mathrm{out}}}{T} \sum_{t=0}^{T-1} Y_{t} = \frac{1}{T} \sum_{t=0}^{T-1} \left(1 - \beta_{\mathrm{out}}^{\,T-t}\right) S_{t}
```

**[Lines 135 to 139](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L135-L139): class scores.** `classify` applies the head: dropout on the readout vector z, then one linear layer with one output per class. `forward` chains `embed` and `classify`, returning logits shaped `[batch, n_classes]`; the softmax comes later, in the loss ([train/loop.py line 141](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L141)) and in prediction ([line 244](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L244)). The linear layer commutes with the time average ([lif\_reference.py lines 29 to 30](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif_reference.py#L29-L30)). So the scores equal the average potential of a non-spiking output layer, with the bias added once.

```latex
\mathrm{logits} = W\, \mathrm{dropout}(z) + b
```

**[Lines 141 to 143](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L141-L143): the firing-rate penalty.** `activity_penalty` returns None when `rate_weight` is 0; otherwise `lif.penalty_of` ([lif.py lines 294 to 305](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L294-L305)) builds it. When training, each LIF layer stores P: the mean, over its N neurons, of (firing rate minus `rate_target`) squared ([lif.py lines 226 to 230](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L226-L230)). `penalty_of` averages P over the model's L LIF layers and multiplies by `rate_weight`. [train/loop.py lines 203 to 204](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L203-L204) add that to the emotion loss, just before `loss.backward()` at [line 400](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L400).

```latex
r_{j} = \frac{\sum_{b,t} m_{b,t}\, S_{b,t,j}}{\sum_{b,t} m_{b,t}}, \qquad P = \frac{1}{N} \sum_{j=1}^{N} \left(r_{j} - r_{\mathrm{target}}\right)^{2}
```

```latex
\mathcal{L} = \mathcal{L}_{\mathrm{emotion}} + w_{\mathrm{rate}} \cdot \frac{1}{L} \sum_{\ell=1}^{L} P_{\ell}
```

**[Lines 145 to 153](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L145-L153): recording, the report, and the finished class.** `record(True)` clears the `Recorder` and switches it on; `record(False)` switches it off. The training loop records only during the final validation pass with the best epoch's weights ([train/loop.py lines 451 to 461](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L451-L461)). `spike_report` ([lif.py lines 308 to 323](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L308-L323)) then gives each layer's firing rate, operations per utterance and an energy estimate for `efficiency.json`. Line 153 returns the class.

**[Lines 156 to 158](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_lif.py#L156-L158): registration.** `register_model` files this builder under the name `"cnn_lif"` with `CNNLIFSettings` as its settings class ([registry.py lines 30 to 41](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/registry.py#L30-L41)). A job whose model section says `"name": "cnn_lif"` is built here ([train/loop.py line 351](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L351)). It receives `n_mels` from the features and the class count from the label set.

### The finalist: settings and shapes

The hybrid finalist is `lif_final` (sweep `cnn_lif_confirm_1`, config `553e54c97626`). Its model settings, from the `settings` field of `artifacts/selections/lif_final/selection.json`:

| Group | Model settings |
| --- | --- |
| Front end | `channels` \[32, 64, 128, 256\]; `convs_per_block` 1; `kernel_size` 3; `pool_freq` 2; `pool_time` 2; `norm` "batch"; `norm_groups` null; `norm_subbands` null; `activation` "relu" |
| To time steps | `freq_reduce` "mean"; `time_pool` 4 |
| Spiking layers | `lif_hidden` \[256\]; `recurrent` false; `lif_norm` "layer"; `lif_dropout` 0.3 |
| LIF neuron | `beta` 0.8; `learn_beta` "none"; `threshold` 0.5; `reset` "zero"; `detach_reset` false; `surrogate` "boxcar"; `surrogate_slope` 1.0 |
| Readout and loss | `readout_beta` 0.5; `rate_weight` 1.0; `rate_target` 0.2; `dropout` 0.3 |

The same file's train settings give `batch_size` 16 and `crop_s` 5.0. IEMOCAP audio is 16,000 Hz ([iemocap.py line 87](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L87)), so each crop is 80,000 samples ([train/loop.py line 364](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L364)). Its features use `n_mels` 64, `win_ms` 25.0 and `hop_ms` 10.0: a 25 ms window every 160 samples. [logmel.py line 111](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L111) then gives 1 + 80000 // 160 = 501 frames.

| Stage | Code (cnn\_lif.py unless named) | Shape |
| --- | --- | --- |
| Waveform crops | [train/data.py lines 121 to 136](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L121-L136) | `[16, 80000]` |
| Log-mel | [logmel.py lines 113 to 120](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L113-L120) | `[16, 64, 501]` |
| Normalisation and SpecAugment | [train/loop.py line 185](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L185) | `[16, 64, 501]` |
| Channel axis added | line 122 | `[16, 1, 64, 501]` |
| Block 1: conv 1 → 32, batch norm, ReLU, max pool 2 × 2 | line 122 | `[16, 32, 32, 250]` |
| Block 2: conv 32 → 64, same layers | line 122 | `[16, 64, 16, 125]` |
| Block 3: conv 64 → 128, same layers | line 122 | `[16, 128, 8, 62]` |
| Block 4: conv 128 → 256, same layers | line 122 | `[16, 256, 4, 31]` |
| Mean over the 4 bands, time moved second | line 124 | `[16, 31, 256]` |
| `time_pool` 4: 8 groups, the last with 3 real steps | line 130 | `[16, 8, 256]` |
| `lif1`: linear 256 → 256 without bias, LayerNorm, LIF, dropout 0.3 | line 132 | `[16, 8, 256]` |
| Readout with `readout_beta` 0.5 | line 133 | `[16, 256]` |
| Head: dropout 0.3, linear 256 → 4 | line 136 | `[16, 4]` |

- Each 3 × 3 convolution has padding 1, so it keeps the size. Only the poolings shrink the time axis, rounding down: 501, 250, 125, 62, 31.
- Spiking steps sit 64 frames apart (16 from the four poolings, times 4), which is 640 ms at a 10 ms hop.
- Evaluation feeds whole utterances in batches of `eval_batch_size` 32. A 10 s utterance gives 1,001 frames, 62 front-end steps and 16 spiking steps.
- The 4 outputs are the classes of `label_set` "four\_merged": neu, hap+exc, ang and sad.
- The model has 455,396 parameters: 388,320 in the front end, 66,048 in `lif1` and 1,028 in the output layer. The leak is fixed (`learn_beta` "none") and stored as a buffer, not a parameter.

### Questions she may ask about this file

- **How many time steps does the LIF layer see?** On a 5.0 s training crop, 501 frames become 31 front-end steps, then 8 after `time_pool` 4 (lines 122 to 130). The steps are 640 ms apart, and longer utterances at evaluation give more steps.
- **Which parts actually spike, and what does that mean for the operation counts?** Only `lif1` spikes; the front end is the CNN's real-valued network (ReLU in the finalist), and the readout has no threshold. Since `lif1` gets real input and has no recurrence, every operation counted for this finalist is a multiply-accumulate.
- **How do gradients pass through a spike?** The forward pass uses a step function, and the backward pass swaps in a surrogate derivative ([lif.py lines 63 to 103](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L63-L103)). Here it is `"boxcar"` with `surrogate_slope` 1.0: 0.5 when the potential is within 0.5 of `threshold`, otherwise 0.
- **Where does the firing-rate penalty enter the loss?** At [train/loop.py lines 203 to 204](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L203-L204), added to the emotion loss, and only in training. With `rate_weight` 1.0 and `rate_target` 0.2, it pulls each of the 256 neurons toward spiking on 20% of real steps.
- **Why scale the readout by `1 - readout_beta`?** So that a neuron firing at a steady rate r reads as about r, whatever the leak, on sequences much longer than 1/(1 - `readout_beta`) steps ([lif\_reference.py lines 26 to 31](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif_reference.py#L26-L31)). On a full 5.0 s training crop (8 steps), a steady rate of 1 reads as 0.875. With `readout_beta` 0 the readout is each neuron's spike count divided by the number of steps.

## The fully spiking network: models/snn.py

The fully spiking network puts LIF neurons after every convolution in place of the activation, so its convolution blocks pass spikes instead of real numbers.

**[Lines 1 to 26](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L1-L26): the module docstring.** It describes convolution blocks like the CNN's, with LIF neurons after every convolution and max pooling of the spikes. Lines 7 to 12 give two ways in: `"frames"`, one log-mel frame per time step, and `"repeat"`, the whole image shown `static_steps` times. Lines 14 to 18 say what follows the blocks: up to two spiking linear layers and the hybrid's readout. Lines 20 to 25 cover batch norm, the activity penalty and operation counting.

**[Lines 27 to 32](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L27-L32): imports.** Unlike the hybrid, this file does not import `conv_front_end`; it builds its own convolutions with neurons attached (lines 96 to 124). It reuses `frames_after` from cnn.py and `check_neuron` from [lif.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py), and again loads no PyTorch at import time.

**[Lines 35 to 38](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L35-L38): encoding settings.** `SNNSettings` is again a frozen dataclass with no defaults. `encoding` is `"frames"` or `"repeat"`; `static_steps` counts the repeats for `"repeat"` and must be None for `"frames"`.

**[Lines 39 to 45](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L39-L45): spiking convolution settings.** These mirror the CNN's front-end fields, but there is no `activation`, because LIF neurons take its place. `conv_norm` is `"batch"`, batch norm on each convolution's current, or `"none"`, which gives the convolutions a bias instead.

**[Lines 46 to 50](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L46-L50): spiking linear layers.** `lif_hidden` lists 0 to 2 spiking linear layers after the convolutions, where the hybrid allows 1 to 3. `recurrent` and `lif_norm` apply to these linear layers only. `lif_dropout` applies to every spiking layer, convolutions included.

**[Lines 51 to 58](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L51-L58): the neuron settings.** These are the hybrid's seven neuron fields, now shared by every convolutional and linear LIF layer. The section on lif.py explains them.

**[Lines 59 to 63](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L59-L63): readout and loss settings.** These are the hybrid's four fields. `rate_target` now applies to each convolution channel as well as each linear unit.

**[Lines 66 to 85](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L66-L85): settings validation.** Lines 67 to 71 tie `static_steps` to the encoding. Lines 72 to 82 repeat the front-end checks by hand; `check_front_end` would need `norm` and `activation`, which these settings lack. Lines 83 to 85 allow at most 2 spiking linear layers and run `check_neuron`. As for the hybrid, [runs/spec.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py#L93-L109) runs this check before any job is queued.

**[Lines 88 to 94](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L88-L94): building the classes with PyTorch.** As in the hybrid, PyTorch is imported only inside `_module_class`. Here both `Neuron` and `SpikingLinear` are used: `Neuron` inside each spiking convolution, `SpikingLinear` after the blocks.

**[Lines 96 to 108](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L96-L108): one spiking convolution layer.** `SpikingConv` bundles a `Conv2d` with padding `k // 2`, which keeps the size, an optional `BatchNorm2d`, and a layer of LIF neurons. The convolution has a bias only when `conv_norm` is `"none"`. `beta_shape` sizes one leak per neuron when `learn_beta` is `"neuron"`: per channel and frequency row with `"frames"`. `fan_out`, `c_out` × k × k, serves only the operation counts.

**[Lines 110 to 115](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L110-L115): the input current.** `current` runs the convolution over all frames at once, then the batch norm if there is one. Multiplying by the frame mask zeroes the current on padded frames. In training, the batch-norm statistics include padded frames, as in the CNN, because the mask comes afterwards.

**[Lines 117 to 124](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L117-L124): counting operations.** While recording, a layer with real input (only the first convolution) adds multiply-accumulates: real input elements × `fan_out`. Every other layer adds accumulates: input spikes on real frames × `fan_out`, so only actual spikes are counted.

**[Lines 126 to 133](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L126-L133): the constructor's first part.** The constructor checks the settings, creates the `Recorder` and stores the encoding. Line 133 builds one `MaxPool2d` with window (`pool_freq`, `pool_time`), reused after every block; it has no weights.

**[Lines 134 to 144](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L134-L144): building the spiking convolution blocks.** Each block gets `convs_per_block` `SpikingConv` layers, named like `conv1.1`, each told how many frequency rows it sees. `real_input` is True only for the very first convolution, the one that reads the log-mel values. `bands` is divided by `pool_freq` after each block, and `width` (line 144) is the last block's channels × the remaining rows.

**[Lines 145 to 150](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L145-L150): spiking linear layers.** Each entry of `lif_hidden` adds a `SpikingLinear`, the same class the hybrid uses. With `"repeat"`, the first one receives per-repeat firing rates, which are real numbers, so it counts multiply-accumulates. With `"frames"`, every linear layer receives spikes.

**[Lines 151 to 155](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L151-L155): dropout, readout settings and the head.** These match the hybrid's lines 96 to 100, plus `dropout_p` for the convolutions' spikes. The head is again dropout and one linear layer to the class scores.

**[Lines 157 to 158](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L157-L158): real frames after the blocks.** This is the hybrid's helper. This file never calls it; both encodings track the count themselves (lines 174 and 202).

**[Lines 160 to 172](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L160-L172): encoding "frames", inside each block.** After line 162 adds a channel axis, time is the frame axis from the first layer on. For each block, line 166 builds the frame mask, and each layer computes its current over all frames at once. The permute at line 169 moves frames to the second axis, so the neurons at line 170 step through them one frame at a time. Dropout follows, and line 172 restores the image layout for the next convolution.

**[Lines 173 to 177](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L173-L177): max pooling the spikes, then steps.** After each block, max pooling keeps each window's largest value (2 × 2 windows for the finalist), so one spike in a window is enough. Line 174 floor-divides the real frame count to match. After the last block, each remaining frame becomes one step: all channels and frequency rows joined into one vector of length `width`.

```latex
s'_{c,f,t} = \max\left(s_{c,2f,2t},\ s_{c,2f+1,2t},\ s_{c,2f,2t+1},\ s_{c,2f+1,2t+1}\right)
```

**[Lines 179 to 184](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L179-L184): encoding "repeat", setup.** The finalist does not use this path. It keeps frames as an image axis and creates a separate time axis: `static_steps` repeats of the same input.

**[Lines 185 to 202](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L185-L202): encoding "repeat", the blocks.** The first convolution's current is computed once and copied to every repeat (lines 189 to 192). Later layers fold the repeats into the batch, convolve, and unfold them (lines 193 to 197). The neurons then integrate along the repeats, so each neuron is one channel at one frequency row and frame. Pooling and the frame count follow as in `"frames"`.

**[Lines 203 to 207](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L203-L207): encoding "repeat", rates per repeat.** For each repeat, each channel and row's spikes are averaged over the utterance's real frames. These rates, `[batch, repeats, channels × rows]`, are the steps for the linear layers, and the mask is all True because every repeat is real.

**[Lines 209 to 219](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L209-L219): embed.** `embed` counts utterances while recording, then runs the chosen encoding on `x.float()`. The steps pass through each `SpikingLinear`, or go straight to the readout if `lif_hidden` is empty. The readout is `lif.readout`, the same call as the hybrid's line 133.

**[Lines 221 to 225](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L221-L225): class scores.** As in the hybrid, the head applies dropout and one linear layer, and `forward` returns logits shaped `[batch, n_classes]`.

**[Lines 227 to 238](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L227-L238): penalty, recording and the finished class.** `activity_penalty` is the hybrid's, but `penalty_of` now averages over every LIF layer, convolutional and linear. A convolution layer's rate is per channel, averaged over batch, real frames and frequency rows ([lif.py lines 226 to 230](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L226-L230)). `record` and `spike_report` work as in the hybrid, and line 238 returns the class.

**[Lines 241 to 243](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L241-L243): registration.** The builder is registered as `"snn"` with `SNNSettings`, so a job whose model section says `"name": "snn"` builds this network.

### How it differs from the hybrid

- **Where spikes start.** The hybrid's convolutions use a real-valued activation (ReLU in its finalist), and only its LIF layers spike. Here every convolution feeds LIF neurons, and with `"frames"` only the first convolution sees real input.
- **What a time step is.** In the hybrid, steps exist only after the convolutions and can be averaged by `time_pool`. Here, with `"frames"`, the neurons of every convolution step through frames (with `"repeat"`, through the repeats), and there is no `time_pool`.
- **Frequency.** The hybrid can average the remaining bands (`"mean"`, as in its finalist). Here they are always kept: each step joins channels × rows.
- **Pooling and normalisation.** Max pooling acts on spikes rather than activation outputs. `conv_norm` normalises each convolution's current before the neurons, where the hybrid's `norm` feeds its activation.
- **Operation counts.** The hybrid's front end counts only multiply-accumulates. Here, with `"frames"`, only the first convolution does, and every later layer counts accumulates; with `"repeat"`, the first linear layer counts multiply-accumulates too.
- **Shared code.** `SpikingLinear`, the readout, the penalty and the recorder come from the same lif.py code, and the head is the same.

### The finalist's settings

The fully spiking finalist is `snn_final` (sweep `snn_confirm_1`, config `396d6ef8c242`). Its model settings, from the `settings` field of `artifacts/selections/snn_final/selection.json`:

| Group | Model settings |
| --- | --- |
| Encoding | `encoding` "frames"; `static_steps` null |
| Spiking convolutions | `channels` \[16, 32, 64\]; `convs_per_block` 2; `kernel_size` 3; `pool_freq` 2; `pool_time` 2; `conv_norm` "batch" |
| Spiking linear layers | `lif_hidden` \[128, 64\]; `recurrent` true; `lif_norm` "batch"; `lif_dropout` 0.0 |
| LIF neuron | `beta` 0.8; `learn_beta` "layer"; `threshold` 1.0; `reset` "subtract"; `detach_reset` false; `surrogate` "boxcar"; `surrogate_slope` 1.0 |
| Readout and loss | `readout_beta` 0.97; `rate_weight` 1.0; `rate_target` 0.2; `dropout` 0.3 |

- Its features use `hop_ms` 20.0 (320 samples) and its crops `crop_s` 4.0, so a training crop gives 1 + 64000 // 320 = 201 frames.
- Block 1's neurons step through 201 frames, block 2's through 100 and block 3's through 50. Each convolution layer holds 1,024 neurons (16 × 64, 32 × 32 or 64 × 16).
- The two recurrent linear layers then run over 25 steps, 160 ms apart; each input step holds 64 × 8 = 512 spike values.
- Its neuron and readout settings match the hybrid finalist's except four. Here `threshold` is 1.0 (hybrid 0.5), `reset` "subtract" (hybrid "zero"), `learn_beta` "layer" (hybrid "none") and `readout_beta` 0.97 (hybrid 0.5).
- It has 166,876 parameters, including one learned leak for each of its 8 LIF layers.

### Questions she may ask about this file

- **How does the audio become spikes?** There is no separate spike encoder. `conv1.1` convolves the normalised log-mel values and feeds the result to its neurons as real current (`real_input` at line 139). The docstring cites Bittar and Garner 2022 for this (line 8). Those neurons, one per channel and frequency row, integrate along the frames and emit the first spikes.
- **What is one time step here?** With `"frames"`, one log-mel frame, 20 ms apart for the finalist. Each block's pooling halves the steps for the next: 201, 100 and 50 on a 4.0 s crop, then 25 for the linear layers.
- **What does max pooling do to spikes?** It outputs 1 if any spike in its 2 × 2 window fired. With `lif_dropout` 0.0, as in the finalist, pooled values stay exactly 0 or 1.
- **What does the penalty average over here?** All 8 LIF layers of the finalist: six convolutional layers with one rate per channel, and two linear layers with one rate per unit. Each layer's mean squared gap to `rate_target` 0.2 is averaged across layers, then multiplied by `rate_weight` 1.0.
- **What does `readout_beta` 0.97 do?** Unrolled as in the hybrid, a spike at step t enters with weight `(1 - 0.97^(T - t)) / T`. On a 25-step crop that is 0.53/25 at the first step and 0.03/25 at the last, so early steps count more.

## The CNN + Transformer and the gender head

Both files were added on 2026-09-26. The CNN + Transformer is "model family 2": the CNN's convolution blocks, then a Transformer encoder across time. The gender head is an optional second output that any model can carry.

### src/affect/models/cnn\_transformer.py

**[src/affect/models/cnn\_transformer.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_transformer.py)**

- [L1-L10](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_transformer.py#L1-L10), docstring: the front end is the CNN's, its output becomes one vector per time step, and a Transformer encoder attends across time. Pooling and classifier are the CNN's. The only fixed number is the position code's base of 10,000.
- [L16-L18](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_transformer.py#L16-L18), imports: seven helpers come from `affect.models.cnn`, so both models share the same convolution blocks, pooling and classifier code.
- [L21-L51](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_transformer.py#L21-L51), `CNNTransformerSettings`: a frozen dataclass with no defaults, in four groups.
  - Front end, as in the CNN: `channels` (one entry per block), `convs_per_block`, `kernel_size`, `pool_freq`, `pool_time`, `norm`, `norm_groups`, `norm_subbands`, `activation`.
  - Bridge: `freq_reduce`, either "mean" (average the remaining frequency bands) or "flatten" (keep every channel of every band).
  - Transformer: `d_model`, `n_heads`, `n_layers`, `ff_dim`, `transformer_dropout`, `transformer_activation` ("relu" or "gelu"), `norm_first` (pre-LN or post-LN), `positional` ("sinusoidal" or "none").
  - Head, as in the CNN: `pooling` ("mean", "max", "meanmax", "stats", "attention"), `classifier_hidden`, `dropout`.
- [L54-L64](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_transformer.py#L54-L64), `check`: runs the CNN's front-end and head checks. It then refuses an `n_heads` that does not divide `d_model`, and `n_layers` or `ff_dim` below 1. It also refuses `transformer_dropout` below 0 or at least 1, and an odd `d_model` with the sinusoidal code.
- [L67-L71](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_transformer.py#L67-L71), `bands_after`: mel bands left after the blocks (`n_mels` floor-divided by `pool_freq` once per block). It sizes the "flatten" input.
- [L74-L83](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_transformer.py#L74-L83), `sinusoidal`: the fixed position code of Vaswani et al., shape \[length, width\]. Even columns hold sines and odd columns cosines, at frequencies 1 / 10000^(2i / width).
- [L86-L144](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_transformer.py#L86-L144), `_module_class`: defines the `nn.Module` inside a function. torch is imported only inside functions, so the settings and checks load without torch.
  - [L91-L116](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_transformer.py#L91-L116), `__init__`: the front end comes from `conv_front_end`. The projection input is the last block's channel count, times the remaining bands for "flatten". `project` is a `Linear` to `d_model`.
  - Same block: the encoder stacks `n_layers` copies of PyTorch's `TransformerEncoderLayer` (`batch_first=True`). With `norm_first` it adds a final `LayerNorm`. `embedding_dim` is `d_model`, doubled for "meanmax" and "stats" pooling.
  - [L118-L119](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_transformer.py#L118-L119), `frames_after_blocks`: real frames left after the time pooling, used to build the padding mask.
  - [L121-L136](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_transformer.py#L121-L136), `embed`: the forward pass up to the pooled summary (shapes below). The key padding mask hides padded steps from attention, and the pooling uses real steps only.
  - [L138-L142](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_transformer.py#L138-L142), `classify` and `forward`: `forward` is `classify(embed(...))`. Keeping the two apart lets a gender head read the same summary.
- [L147-L149](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn_transformer.py#L147-L149): registers the builder as model "cnn\_transformer". A sweep selects it with `"model": {"name": "cnn_transformer", ...}`.

Shapes inside `embed`:

```
x              [batch, n_mels, frames]
front end      [batch, channels, bands, steps]     steps: frames floor-divided by pool_time once per block
"mean"         [batch, steps, channels]            average over the remaining bands
"flatten"      [batch, steps, channels x bands]
project        [batch, steps, d_model]             plus the sinusoidal code when positional is "sinusoidal"
encoder        [batch, steps, d_model]             src_key_padding_mask is True on padded steps
masked_pool    [batch, embedding_dim]
classifier     [batch, n_classes]
```

The shared helpers live in cnn.py: [`conv_front_end`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L165-L187), [`frames_after`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L203-L210), [`masked_pool`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L98-L121), [`pooled_width`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L124-L125), [`classifier_head`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L190-L200), [`check_front_end`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L50-L83) and [`check_head`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L86-L90).

### src/affect/models/heads.py

**[src/affect/models/heads.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/heads.py)**

- [L1-L8](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/heads.py#L1-L8), docstring: the head is dropout plus one linear layer, from the model's embedding to two logits (female, male). "multitask" trains it like the emotion output. "adversarial" puts a gradient reversal (Ganin and Lempitsky 2015) in front of it.
- [L10](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/heads.py#L10), `AUX_KINDS`: `("multitask", "adversarial")`.
- [L17-L24](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/heads.py#L17-L24), `_Reverse`: an autograd function. Forward returns its input unchanged; backward returns the gradient negated.
- [L26-L35](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/heads.py#L26-L35), `AuxHead`: `Dropout`, then `Linear(width, n_out)`. With `reverse` set, the embedding first passes through `_Reverse`.
- [L40-L43](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/heads.py#L40-L43), `build_aux_head`: refuses any kind outside `AUX_KINDS`. The reversal is on only for "adversarial".

How training uses the head:

- [train/settings.py L30-L35](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L30-L35): `train.aux_gender` is "none", "multitask" or "adversarial", and `train.aux_gender_weight` weights the gender loss. The comment says emotion alone picks the epoch. [L68-L72](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L68-L72) requires weight 0 with "none" and above 0 otherwise.
- [train/loop.py L147-L157](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L147-L157): `make_aux_head` builds the head with the model's `embedding_dim`, two outputs ([`GENDERS`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L204) is `("F", "M")`) and the model's classifier dropout. `trainable_parameters` adds its weights to the optimizer.
- [train/loop.py L194-L208](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L194-L208): with a head, the step calls `embed` once, then `classify` and the head on that summary. The loss is the emotion loss (plus a spiking model's activity penalty) plus `aux_gender_weight` times the gender cross-entropy.
- So the weight scales the gradient that reaches the shared layers in both modes. "adversarial" only flips its sign; the head's own weights learn gender either way.
- [train/loop.py L341-L344](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L341-L344): gender targets are one-hot from the speaker's gender, marked "never shuffled". [L433-L439](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L433-L439): the best epoch is chosen on emotion validation UA, and the head's weights are saved with it.
- [train/loop.py L486-L488](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L486-L488) with [`gender_report`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L292-L305): each run records the head's accuracy, recall per gender and their mean under "aux\_gender" in `metrics_val.json` and `metrics_test.json`.

Where they are used and tested:

- [sweeps/cnn\_gender\_aux\_1.json](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/sweeps/cnn_gender_aux_1.json) tries multitask weights 0.1, 0.3 and 1.0 and adversarial weights 0.03, 0.1 and 0.3. Five sweep files use the CNN + Transformer as their base model (`sweeps/cnn_transformer_*.json`).
- [tests/test\_torch\_parts.py L304-L343](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_torch_parts.py#L304-L343) checks the model's outputs, gradients, padding mask, position code and refused settings. [L345-L364](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_torch_parts.py#L345-L364) checks that the reversal leaves the forward pass alone and negates the gradient.

## Training: train/settings.py, train/data.py, train/loop.py

This section follows one training job from its settings to the files it writes, with every link pointing at the exact lines on GitHub.

### train/settings.py

**[Lines 1 to 8](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L1-L8): the module note.** Five fields were added later: aux\_gender, aux\_gender\_weight, eval\_segment\_s and eval\_segment\_hop\_s on 2026-09-26, and eval\_train on 2026-10-03. Settings written earlier lack them, and in those settings a missing field means what the loop did before: no gender head, whole-utterance evaluation and no training pass.

**[Lines 13 to 14](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L13-L14): one frozen dataclass.** TrainSettings is frozen, so no code can change a setting once it is validated. No field has a default ([affect/settings.py lines 99 to 101](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/settings.py#L99-L101)), so every sweep file spells out every value, apart from fields added after it was written (lines 48 to 50).

**[Line 15](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L15): `epochs`.** The most passes over the training sessions a job may make; early stopping can end it sooner. Finalists: 100 (threeblock\_final, cnn\_final\_3) or 60 (lif\_final, snn\_final). Baseline: 60.

**[Line 16](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L16): `batch_size`.** Training utterances per optimizer step. The last incomplete batch of each epoch is dropped (loop.py line 357). Finalists: 16. Baseline: 32.

**[Line 17](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L17): `lr`.** The peak learning rate of AdamW, reached when warmup ends. Every record uses 0.001, except cnn\_final\_3 with 0.0003.

**[Line 18](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L18): `weight_decay`.** AdamW's weight decay: every step also shrinks each parameter slightly toward zero, which discourages large weights. Values: 0.0001 (threeblock\_final, lif\_final, baseline), 0.01 (snn\_final) and 0.003 (cnn\_final\_3).

**[Line 19](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L19): `warmup_epochs`.** Over this many epochs the learning rate climbs linearly up to lr (loop.py lines 41 to 42). All five records use 2.

**[Line 20](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L20): `schedule`.** After warmup, "cosine" lowers the rate along a half cosine that reaches 0 at the end of the last allowed epoch. "constant" holds it at lr. cnn\_final\_3 uses constant; the other four records use cosine.

**[Line 21](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L21): `early_stopping_patience`.** The job stops once this many epochs pass without a higher validation UA. Finalists: 20. Baseline: 10.

**[Line 22](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L22): `crop_s`.** The length in seconds of the random training crop; shorter utterances are zero-padded and the padding is masked. Values: 10.0 (threeblock\_final, cnn\_final\_3), 5.0 (lif\_final, baseline) and 4.0 (snn\_final).

**[Line 23](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L23): `class_balance`.** "none"; "weighted\_loss", which weights each class's loss term by its rarity; or "balanced\_sampler", which draws rarer classes more often. lif\_final, snn\_final and cnn\_final\_3 use weighted\_loss. threeblock\_final and the baseline use none.

**[Lines 24 to 25](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L24-L25): `labels` and `label_smoothing`.** "hard" trains on one class per utterance; "soft" trains on the spread of the evaluators' votes over the four classes. Label smoothing mixes the target with a uniform distribution by this weight. cnn\_final\_3 uses soft labels, lif\_final and snn\_final use smoothing 0.1, and the rest use 0.0.

**[Line 26](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L26): `precision`.** "bf16" runs the model's forward pass in bfloat16, a 16-bit number format, inside PyTorch's autocast. "fp32" keeps 32-bit floats throughout. All five records use bf16.

**[Line 27](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L27): `eval_batch_size`.** Whole utterances per batch during evaluation (segments, when segment evaluation is on). All five records use 32.

**[Line 28](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L28): `control`.** "shuffle\_train\_labels" scrambles the training labels as a leak test, where validation UA should fall to chance. All five records use "none".

**[Line 29](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L29): `max_batches_per_epoch`.** A cap on steps per epoch, used only by the smoke test. All five records hold null (None).

**[Lines 30 to 35](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L30-L35): `aux_gender` and `aux_gender_weight`.** An optional second output predicts the speaker's gender from the model's pooled features. "multitask" lets the shared layers learn gender; "adversarial" reverses the gradient so they unlearn it. Emotion alone picks the epoch. All five records: none, with weight 0.0 (the baseline lacks both fields, which means the same).

**[Lines 36 to 40](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L36-L40): `eval_segment_s` and `eval_segment_hop_s`.** None means validation and test see each utterance whole. A length in seconds instead cuts each utterance into segments, one every hop seconds, and averages their class probabilities. All five records: None (the baseline lacks both fields).

**[Lines 41 to 46](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L41-L46): `eval_train`.** "every\_epoch" predicts the training utterances after each validation pass, the same way as validation, and records their UA, WA, macro-F1 and mean NLL. It changes no weight, no chosen epoch and no validation or test number. threeblock\_final, lif\_final and snn\_final use every\_epoch; cnn\_final\_3 and the baseline predate the field, so for them it is "none".

**[Lines 48 to 50](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L48-L50): `ADDED_FIELDS`.** The value each added field takes when an older settings dict lacks it. Validation fills these in ([affect/settings.py lines 226 to 239](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/settings.py#L226-L239)). Hashing leaves them out while they hold those values ([lines 256 to 272](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/settings.py#L256-L272)), so old runs keep their names.

**[Lines 53 to 79](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/settings.py#L53-L79): `check`.** Range checks, for example: batch\_size at least 2, warmup\_epochs below epochs, and label\_smoothing at least 0 but below 1. A gender weight must be above 0 exactly when a head is on. The checks run when a sweep is expanded ([sweep/design.py line 74](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L74)) and again at the start of each job (loop.py line 315).

### Training settings of the finalists and the baseline

Each column is the `settings.train` section of one selection record under artifacts/selections. The first row comes from `settings.model.name`. "absent" means the record's settings were written before the field existed; validation then fills in the value in parentheses.

| Field | threeblock\_final | lif\_final | snn\_final | cnn\_final\_3 | cnn\_base\_session\_out\_2 |
| --- | --- | --- | --- | --- | --- |
| model (model section) | cnn | cnn\_lif | snn | cnn | cnn |
| epochs | 100 | 60 | 60 | 100 | 60 |
| batch\_size | 16 | 16 | 16 | 16 | 32 |
| lr | 0.001 | 0.001 | 0.001 | 0.0003 | 0.001 |
| weight\_decay | 0.0001 | 0.0001 | 0.01 | 0.003 | 0.0001 |
| warmup\_epochs | 2 | 2 | 2 | 2 | 2 |
| schedule | cosine | cosine | cosine | constant | cosine |
| early\_stopping\_patience | 20 | 20 | 20 | 20 | 10 |
| crop\_s | 10.0 | 5.0 | 4.0 | 10.0 | 5.0 |
| class\_balance | none | weighted\_loss | weighted\_loss | weighted\_loss | none |
| labels | hard | hard | hard | soft | hard |
| label\_smoothing | 0.0 | 0.1 | 0.1 | 0.0 | 0.0 |
| precision | bf16 | bf16 | bf16 | bf16 | bf16 |
| eval\_batch\_size | 32 | 32 | 32 | 32 | 32 |
| control | none | none | none | none | none |
| max\_batches\_per\_epoch | null | null | null | null | null |
| aux\_gender | none | none | none | none | absent (none) |
| aux\_gender\_weight | 0.0 | 0.0 | 0.0 | 0.0 | absent (0.0) |
| eval\_segment\_s | null | null | null | null | absent (None) |
| eval\_segment\_hop\_s | null | null | null | null | absent (None) |
| eval\_train | every\_epoch | every\_epoch | every\_epoch | absent (none) | absent (none) |

Each finalist record also holds a `baseline` field: the configuration hash of the baseline its confirmation sweep ran beside it. cnn\_final\_3 names cf0495a296e9, the configuration of cnn\_base\_session\_out\_2. threeblock\_final names 66d8d74d5623, the configuration of threeblock\_session\_out\_2. That record's train values match cnn\_base\_session\_out\_2's, but it lists the added fields, with eval\_train every\_epoch. lif\_final and snn\_final name their own sweeps' baselines (abd976bf64e5 and 88dff43c3eed), which have no selection record.

### train/data.py

**[Lines 1 to 6](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L1-L6): the module note.** A job's dataset is loaded once per worker process and reused by later jobs. All audio is held in memory as 16-bit samples, about 0.8 GB for the four IEMOCAP classes. Features are computed on the GPU, batch by batch.

**[Lines 7 to 17](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L7-L17): imports.** `wave` reads WAV files and numpy holds the samples. The rest are project helpers for paths, speed perturbation, hashing, settings and the dataset registry.

**[Lines 20 to 33](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L20-L33): `Prepared`, what a job receives.** The utterance list and a map from utterance ID to row number. Per row: the audio as an int16 array, the hard label index, the soft vote distribution and the kind (impro or script). Also the class names, the folds, and hashes of the folds and of the data manifest.

**[Lines 36 to 37](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L36-L37): two caches.** `_PREPARED` keeps each prepared dataset. `_AUDIO` keeps each set of audio, so jobs of different split schemes over the same utterances share one copy.

**[Lines 40 to 48](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L40-L48): `read_int16`.** Opens one WAV file and refuses anything but mono, 16-bit, uncompressed audio. It reads every frame as a little-endian int16 and checks the count against the utterance table.

**[Lines 51 to 60](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L51-L60): `_audio`.** Reads the audio of a list of utterances once per process, keyed by the data folder and each utterance's ID, path and length.

**[Lines 63 to 88](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L63-L88): `prepare`.** Returns the cached record, or builds it: utterances, class order, audio, label indices and soft targets. It requires a single sample rate, checks that every soft target sums to 1, builds the folds and hashes the manifest.

**[Lines 91 to 100](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L91-L100): speaker helpers.** `speaker_index` gives each row its speaker's position in a sorted list, for per-speaker normalization. `rows_per_speaker` takes the first k rows of each speaker; only the smoke test uses it.

**[Lines 103 to 110](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L103-L110): `train_order`, one epoch's order.** By default, a fresh random permutation: each training utterance appears once, in a new order every epoch (the loop then drops the last incomplete batch, loop.py line 357). With "balanced\_sampler", n positions are drawn with replacement, each with probability proportional to one over its class count. The draws come from the generator the loop passes in.

**[Lines 113 to 118](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L113-L118): `class_weights`.** One weight per class, from the training labels; a class with no training example raises an error.

```latex
w_k = \frac{n}{K \, n_k}
```

Here n is the number of training utterances, K the number of classes and n\_k the count of class k. A class of average size gets weight 1, and a rarer class gets more.

**[Lines 121 to 136](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L121-L136): `crop_arrays`, how a training crop is drawn and padded.** For an utterance longer than the crop, one start is drawn uniformly over every possible position, and crop\_len samples are copied. A shorter utterance is copied whole and the rest of its row stays 0. `valid` records the real length, and dividing by 32768 scales the samples to the range from -1 to just under 1. Exactly one number is drawn per long utterance, in order, so the seed fixes every crop.

**[Lines 139 to 152](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L139-L152): `crop_batch` and `training_waves`.** `crop_batch` fetches the rows' arrays and crops them. `training_waves` is what the loop calls: with speed\_factors `[1.0]` (set in the four finalists, and absent, which means `[1.0]`, in the baseline), it is exactly `crop_batch`. Otherwise each row is first resampled at a speed drawn from the separate augmentation stream.

**[Lines 155 to 166](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L155-L166): `eval_batches`, whole utterances for evaluation.** Utterances are sorted longest first and grouped batch\_size at a time. Each batch is zero-padded to its longest member, so sorting keeps the padding small. It yields positions, waves and real lengths, so the model can ignore the padding.

**[Lines 169 to 180](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L169-L180): `segment_starts`.** A segment starts every hop samples while a whole segment fits, plus one last segment ending at the final sample if needed. An utterance no longer than one segment is a single segment.

**[Lines 183 to 201](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L183-L201): `segment_batches`.** Lists every segment of every utterance, sorts them longest first and batches them with padding. Each segment carries its utterance's position, so `predict` can average them. No record here uses segments.

**[Lines 204 to 216](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L204-L216): `GENDERS` and `gender_index`.** Gives 0 for a female and 1 for a male speaker, from each utterance's speaker\_gender field. Only a job with a gender head uses it.

### train/loop.py

**[Lines 308 to 501](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L308-L501): the main path.** `train_job` runs one job. It is followed here in the order it runs, stepping into `Step.loss`, `predict` and `evaluate` where they are called. The remaining blocks come at the end.

#### Setting up the job

**[Lines 308 to 309](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L308-L309): `train_job`.** It takes the job spec (settings, config hash, fold, seed, run ID), the GPU device and a `log` function. It returns the content of six run-folder files, keyed by file name, and the worker writes them to disk.

**[Lines 310 to 312](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L310-L312): torch and the start time.** torch is imported inside the function, as everywhere in this file. `started` records the UTC start time for provenance.json.

**[Line 313](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L313): validate the settings.** `validate_job` ([runs/spec.py lines 43 to 55](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py#L43-L55)) checks the five sections (data, features, augment, model, train) against their dataclasses. A missing, unknown or mistyped field raises an error.

**[Line 314](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L314): load the data.** `prepare` (data.py lines 63 to 88) returns the in-memory dataset, reused if this worker loaded it before.

**[Line 315](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L315): cross-section checks.** `check_job` runs the checks that span sections, including `check` of the training settings.

**[Lines 316 to 317](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L316-L317): the hash check.** The validated settings must hash to the config\_hash stored in the job. If the job file was edited after it was enqueued, the job refuses to run.

**[Lines 318 to 320](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L318-L320): find the fold.** The spec names a fold such as test\_S1; an unknown name raises an error.

**[Lines 321 to 324](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L321-L324): split the rows.** `assign` lists the utterance IDs of the train, validation and test roles, which become arrays of row numbers. In leave\_one\_session\_out, the validation session is the one after the test session, and the other three train ([data/splits.py lines 3 to 5](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L3-L5)). `assign` also returns train\_final, train plus validation, for the retraining on four sessions that data/splits.py line 6 describes; `train_job` never uses it, so no job of this loop retrains.

**[Line 325](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L325): the settings and the seed.** `ts` is the train section. The seed comes from the spec, so one job is one configuration, one fold and one seed.

**[Line 326](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L326): seed PyTorch.** `torch.manual_seed` fixes PyTorch's global random generators, on the CPU and the GPU. The model's initial weights (line 351, built on the CPU) and its dropout masks (on the GPU) draw from them.

**[Lines 327 to 328](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L327-L328): seed the order, crop and mask streams.** `rng` is a numpy generator for each epoch's order (line 388) and each crop's start (line 395). `mask_generator` is a separate CPU torch generator for the SpecAugment masks.

**[Line 329](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L329): the augmentation stream.** `augmentation_rng` (lines 34 to 37) seeds numpy with `[seed, 20260927]` for speed perturbation and mixup. Because it is a separate stream, they take no numbers from the order, crop and mask streams, so with both off, as in all five records, a job makes exactly the draws it made before they existed. Speed perturbation still changes the crops when on, because a resampled utterance has a different length ([augment/speed.py line 6](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/speed.py#L6)).

**[analysis/compare.py lines 4 to 6](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/compare.py#L4-L6): determinism.** The code seeds every generator, but sets no cuDNN deterministic flag anywhere under src/. This module states that repeated runs were identical.

**[Line 330](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L330): the class count.** K, the number of classes: 4 for the finalists' four\_merged label set (neu, hap+exc, ang, sad).

**[Line 332](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L332): the feature extractor.** Builds the log-mel extractor on the GPU. It turns each batch of waves into log-mel spectrograms: one row per mel band, one column per frame.

**[Line 333](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L333): the training targets.** With "soft" labels each training row's target is its vote distribution; with "hard", a one-hot vector of its label. Only training rows are kept, in train\_rows order.

**[Line 334](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L334): the training labels.** The hard class index of each training row, used for class weights and the balanced sampler.

**[Lines 335 to 337](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L335-L337): the leak test.** With control "shuffle\_train\_labels", targets and labels are permuted together by a generator seeded with `seed + 1_000_003`. All five records use "none".

**[Lines 338 to 340](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L338-L340): label smoothing.** With ε = label\_smoothing and K classes, each target becomes:

```latex
y'_{k} = (1 - \varepsilon)\, y_{k} + \frac{\varepsilon}{K}
```

For lif\_final and snn\_final (ε = 0.1, K = 4), a one-hot target becomes 0.925 for the labeled class and 0.025 for each other class. The result is stored as float32.

**[Lines 341 to 344](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L341-L344): gender targets.** Only with a gender head: a one-hot female or male target per training row. They are never shuffled, even in the leak test.

**[Lines 345 to 347](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L345-L347): class weights.** With "weighted\_loss", the weights of data.py lines 113 to 118 are moved to the GPU; otherwise `weights` stays None.

**[Lines 348 to 349](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L348-L349): the normalizer.** "speaker" normalization takes statistics from every row's audio, and "global" from training rows only. All five records use "utterance", which needs no statistics: each utterance is scaled by its own (lines 49 to 82).

**[Line 351](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L351): the model.** Builds the model named in the model section (cnn, cnn\_lif or snn here) for the number of mel bands and classes, on the GPU. Its random initial weights come from the generator seeded at line 326.

**[Line 352](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L352): the gender head.** `make_aux_head` (lines 147 to 153) returns None unless aux\_gender asks for a head.

**[Line 353](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L353): the parameter count.** It goes into the log and efficiency.json; a gender head is counted separately (line 489).

**[Line 354](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L354): the optimizer.** AdamW over all trainable parameters, with lr and weight\_decay from the settings. They form one group, so weight decay applies to every parameter, biases included. The other AdamW arguments keep PyTorch's defaults.

**[Line 355](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L355): the step object.** `Step` (lines 168 to 208) bundles everything that turns one batch into a loss.

**[Line 356](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L356): evaluation segments.** `segment_samples` (lines 160 to 165) returns None for whole-utterance evaluation, as in all five records.

**[Lines 357 to 359](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L357-L359): steps per epoch.** The number of full batches in the training set; lines 358 to 359 cap it in the smoke test only. Fold test\_S1 has 3423 training utterances, so batch 16 gives 213 steps and leaves 15 utterances out of each epoch.

**[Lines 360 to 363](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L360-L363): the learning-rate schedule.** Warmup and total lengths are counted in optimizer steps. `LambdaLR` sets the rate to lr times `lr_factor` (lines 40 to 46) of the step number s, counted from 0:

```latex
f(s) =
\begin{cases}
\dfrac{s + 1}{W} & \text{if } s < W \\[6pt]
1 & \text{otherwise, constant schedule} \\[6pt]
\dfrac{1}{2}\left(1 + \cos\left(\pi \min\left(\dfrac{s - W}{\max(1,\, T - W)},\ 1\right)\right)\right) & \text{otherwise, cosine schedule}
\end{cases}
```

W is the warmup length (warmup\_epochs times steps per epoch) and T the total (epochs times steps per epoch). The cosine reaches 0 only at step T, so a job that stops early never gets there.

**[Line 364](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L364): crop length in samples.** crop\_s times the sample rate, rounded. At IEMOCAP's 16000 Hz ([data/iemocap.py line 87](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L87)), 10.0 s is 160000 samples and 4.0 s is 64000.

**[Lines 365 to 366](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L365-L366): reset the GPU memory peak.** The peak\_gpu\_mib in efficiency.json then reflects this job alone.

**[Lines 368 to 379](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L368-L379): the header line of log.txt.** Run ID, fold and validation session, seed, set sizes, parameter count, steps per epoch, and any optional feature in use. One lif\_final run logged `run 553e54c97626_test_S1_seed6: fold test_S1 (val S2), seed 6, train 3423, val 1023, test 1085, 455396 parameters, 213 steps/epoch`.

**[Lines 380 to 382](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L380-L382): the selection state.** `best_ua` starts at -1.0, below any real UA, so epoch 1 always becomes the first best. `best_state` will hold a copy of the best weights, and `curves` one record per epoch. Line 382 starts the clock for the total training time (line 446).

#### The epoch loop and the training step

**[Line 383](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L383): the epoch loop.** Epochs run from 1 to `epochs`, unless early stopping breaks out first.

**[Lines 384 to 386](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L384-L386): training mode.** `model.train()` turns dropout on and lets batch norm use and update batch statistics. It is needed every epoch, because `predict` switches the model to eval mode.

**[Lines 387 to 391](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L387-L391): the epoch's order and loss sums.** The epoch timer starts, and `train_order` (data.py lines 103 to 110) draws this epoch's order from `rng`. The loss sums stay on the GPU as tensors and are read only after the last batch.

**[Lines 392 to 394](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L392-L394): pick a batch.** Batch b takes the next batch\_size positions of the order and maps them to dataset rows.

**[Line 395](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L395): cut the crops.** `training_waves` (data.py lines 144 to 152) returns the batch's cropped, zero-padded waves and their real sample counts.

**[Lines 396 to 398](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L396-L398): forward pass and loss.** `step.loss` returns the total loss to minimise, the emotion loss, and the gender loss (None without a head). Its lines come next.

**[Lines 168 to 182](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L168-L182): what `Step` holds.** The augment and train settings, model, gender head, feature extractor, normalizer, class weights and device. The smoke test uses the same class, so it runs exactly the code a real job runs.

**[Lines 183 to 184](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L183-L184): log-mel and real frames.** The waves move to the GPU and become log-mel spectrograms, shaped batch by mel bands by frames. A crop of n real samples has 1 + n // hop real frames ([features/logmel.py line 111](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L111)).

**[Line 185](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L185): normalize, then mask.** The normalizer rescales each mel band and sets padded frames to 0. SpecAugment ([augment/specaugment.py lines 24 to 41](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/augment/specaugment.py#L24-L41)) then zeroes random bands and frame spans, drawn from `mask_generator`. The four finalists use 2 frequency and 2 time masks; the baseline record sets every mask field to 0.

**[Lines 186 to 187](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L186-L187): targets to the GPU.** The emotion targets move to the GPU, and so do the gender targets when a head exists.

**[Lines 188 to 192](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L188-L192): mixup.** With mixup\_alpha above 0, each example is blended with a partner from the batch, and so are its targets. mixup\_alpha is 0.0 in the finalists and absent (0.0) in the baseline, so this is skipped.

**[Line 193](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L193): bf16 autocast.** With precision "bf16", PyTorch runs the operations inside this block in bfloat16 where it can. Feature extraction above and the loss below stay in 32-bit floats.

**[Lines 194 to 199](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L194-L199): the forward pass.** Without a gender head, the model maps the batch to logits: one raw score per class per utterance. With a head, the pooled embedding is computed once and feeds both the emotion classifier and the gender head.

**[Line 200](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L200): the emotion loss.** `soft_cross_entropy` (lines 138 to 144) takes the log-softmax of the logits in float32 and multiplies it by the targets and class weights. It then sums over classes and averages over the batch:

```latex
\mathcal{L}_{\mathrm{emo}} = \frac{1}{B} \sum_{i=1}^{B} \sum_{k=1}^{K} -\, w_k \, y'_{ik} \, \log \frac{e^{z_{ik}}}{\sum_{j=1}^{K} e^{z_{ij}}}
```

B is the batch size, z\_i the logits of utterance i, y'\_i its target after smoothing, and w\_k the class weight (1 without weighting). The sum is divided by B, not by the total weight.

**[Lines 201 to 204](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L201-L204): the LIF firing-rate penalty.** Only the spiking models (cnn\_lif, snn) have `activity_penalty`; for other models, `penalty` is None. In training, LIF layer l stores the mean squared gap between each unit's firing rate r\_lj and rate\_target r\* ([models/lif.py lines 225 to 230](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L225-L230)). `penalty_of` averages the L layers and multiplies by rate\_weight λ\_r ([models/lif.py lines 294 to 305](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L294-L305)).

```latex
P = \lambda_r \cdot \frac{1}{L} \sum_{l=1}^{L} \frac{1}{N_l} \sum_{j=1}^{N_l} \left( r_{lj} - r^{*} \right)^{2}
```

N\_l counts layer l's units; lif\_final and snn\_final both set rate\_weight 1.0 and rate\_target 0.2 in their model sections.

**[Lines 205 to 206](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L205-L206): no gender head.** The step returns the total (the emotion loss, plus the penalty for spiking models), the emotion loss and None.

**[Lines 207 to 208](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L207-L208): the gender loss.** With a head, the gender cross-entropy, without class weights, is scaled by aux\_gender\_weight and added. In "adversarial" mode the head flips the sign of the gradient reaching the shared layers ([models/heads.py lines 17 to 35](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/heads.py#L17-L35)).

**[Lines 200 to 208](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L200-L208): the full loss.**

```latex
\mathcal{L} = \mathcal{L}_{\mathrm{emo}} + P + \lambda_g \, \mathcal{L}_{\mathrm{gender}}
```

P exists only for spiking models with rate\_weight above 0, and the gender term only with a gender head (λ\_g = aux\_gender\_weight). threeblock\_final and cnn\_final\_3 train on the emotion loss alone. lif\_final and snn\_final train on the emotion loss plus P.

**[Line 399](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L399): clear old gradients.** PyTorch adds each new gradient to the stored one, so the old ones are cleared first. `set_to_none=True` sets them to None instead of filling them with zeros.

**[Line 400](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L400): the backward pass.** Backpropagation computes the gradient of the loss with respect to every trainable parameter.

**[Line 401](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L401): the update.** AdamW moves each parameter using its gradient, running averages of past gradients, and the weight decay.

**[Line 402](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L402): advance the schedule.** The scheduler steps once per optimizer step, so the learning rate changes within an epoch, not only between epochs.

**[Lines 403 to 406](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L403-L406): accumulate the losses.** The batch loss joins the epoch sum, detached so no gradient history is kept. With a gender head, the emotion and gender parts are summed separately.

**[Lines 407 to 409](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L407-L409): the epoch's training loss.** The mean batch loss; for lif\_final and snn\_final it includes the rate penalty. A NaN or infinite value stops the job with an error.

**[Line 410](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L410): the epoch's training time.** It excludes evaluation.

#### Validation every epoch

**[Lines 411 to 412](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L411-L412): predict the validation session.** `predict` runs the current weights over every validation utterance. Its lines come next.

**[Lines 218 to 223](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L218-L223): what `predict` takes.** The model, feature extractor, normalizer, audio, the rows to predict, batch size, device and precision. Segments and a gender head are optional.

**[Lines 224 to 226](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L224-L226): eval mode.** Dropout turns off, and batch norm uses its running statistics instead of the batch's. A gender head, when one is passed, switches too.

**[Lines 227 to 230](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L227-L230): accumulators and batches.** `sums` will hold each row's class probabilities, and `counts` how many pieces each row received. Batches are whole utterances (data.py lines 155 to 166) or segments (data.py lines 183 to 201).

**[Lines 231 to 233](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L231-L233): timer and no\_grad.** The timer covers the whole pass. Inside `torch.no_grad()`, PyTorch records no gradient history, which saves memory and time; no weight changes here.

**[Lines 234 to 236](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L234-L236): features as in training.** Log-mel, real frame counts and normalization, exactly as in `Step.loss`, but with no SpecAugment and no mixup.

**[Lines 237 to 243](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L237-L243): the forward pass.** It runs under the same autocast as training and gives emotion logits, plus gender logits when a head is passed.

**[Line 244](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L244): probabilities.** Softmax in float32 turns the logits into class probabilities that sum to 1, copied to the CPU as numpy.

**[Lines 245 to 250](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L245-L250): allocate on the first batch.** The arrays are sized from the first batch's number of classes.

**[Lines 251 to 259](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L251-L259): store or add.** For whole utterances, each row's probabilities are written once. For segments, each segment's probabilities are added to its utterance's row. Either way, the row's count goes up by one.

**[Lines 260 to 262](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L260-L262): stop the clock.** GPU work runs asynchronously, so the code waits for the GPU to finish before reading the timer.

**[Lines 263 to 269](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L263-L269): finish.** For segments, each row's sum is divided by its count, the mean over its S segments:

```latex
\hat{p}(u) = \frac{1}{S} \sum_{s=1}^{S} p(\mathrm{segment}_s)
```

For whole utterances, every row must have been seen exactly once, or the job raises an error. The result holds the probabilities, any gender probabilities and the seconds taken.

**[Line 413](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L413): score the predictions.** `evaluate` turns the probabilities into metrics; its lines come next.

**[Lines 283 to 285](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L283-L285): truth and prediction.** `evaluate` takes the true hard labels of the rows. The predicted class is the one with the highest probability.

**[Line 286](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L286): the report.** `report` ([eval/metrics.py lines 66 to 80](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/metrics.py#L66-L80)) builds the confusion matrix, true classes as rows, and computes every metric from it. That includes UA, WA, macro-F1, and per-class recall, precision and F1.

```latex
\mathrm{UA} = \frac{1}{K} \sum_{k=1}^{K} \frac{C_{kk}}{\sum_{j=1}^{K} C_{kj}}
\qquad
\mathrm{WA} = \frac{\sum_{k=1}^{K} C_{kk}}{\sum_{k=1}^{K} \sum_{j=1}^{K} C_{kj}}
```

C\_kj counts utterances of true class k predicted as class j, and WA is plain accuracy. UA is the mean of the per-class recalls ([eval/metrics.py lines 54 to 55](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/metrics.py#L54-L55)), so every class counts equally whatever its size.

**[Line 287](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L287): by kind.** The same report for improvised and scripted utterances separately.

**[Lines 288 to 289](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L288-L289): mean NLL.** The average negative log-probability given to the true class (formula below), with probabilities clipped at 1e-12 so the log stays finite. Line 289 returns the full report.

```latex
\mathrm{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \log \max\left(\hat{p}_{i,\,y_i},\ 10^{-12}\right)
```

N is the number of utterances, and y\_i is the true class of utterance i.

**[Lines 414 to 420](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L414-L420): the training pass.** With eval\_train "every\_epoch", the training utterances are predicted like validation: eval mode, whole utterances, no augmentation, true labels. It uses the weights just validated and changes no weight, statistic or random state. This is how training UA is measured.

**[Lines 421 to 424](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L421-L424): the epoch record.** Training loss, validation UA, WA, macro-F1 and mean NLL, the learning rate and the two times. The rate is read after the epoch's last scheduler step, so it is the one the next step uses. One lif\_final run recorded 0.0005023474178403757 after epoch 1, still in warmup, and 0.001 after epoch 2.

**[Lines 425 to 431](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L425-L431): optional columns.** With a gender head, the separate emotion and gender losses. With eval\_train, the training UA, WA, macro-F1, mean NLL and time.

**[Line 432](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L432): store the record.** It joins `curves`, which goes into metrics\_val.json.

#### Keeping the best epoch and stopping early

**[Line 433](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L433): the selection rule.** An epoch improves only if its validation UA is strictly higher than the best so far, so a tie keeps the earlier epoch.

**[Lines 434 to 439](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L434-L439): keep the best weights.** The epoch and its UA are stored, and every tensor of the model's state is cloned into `best_state`. That state holds batch-norm running statistics as well as weights. The epoch's training report and gender-head state are kept too.

**[Lines 440 to 442](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L440-L442): the epoch line of log.txt.** Loss, training UA when measured, validation UA and WA, and the two times. The line ends with "best" when the epoch improved.

**[Lines 443 to 445](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L443-L445): early stopping.** The loop ends once the gap since the best epoch reaches the patience:

```latex
\text{stop after epoch } e \quad \text{if} \quad e - e_{\mathrm{best}} \ge \text{patience}
```

With patience 20, one lif\_final run on test\_S1 with seed 6 kept epoch 30 and stopped after epoch 50. `stopped_early` records that the condition was met, even at the last allowed epoch, where the loop would have ended anyway: two snn\_final runs (test\_S3 with seed 9, test\_S5 with seed 6) kept epoch 40, ran all 60 epochs and still record it as true.

**[Line 446](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L446): total training time.** The time since line 382, so unlike each epoch's train\_seconds it includes every validation pass and, with eval\_train, every training pass. It goes into efficiency.json.

#### After training: best weights, test pass, outputs

**[Lines 448 to 450](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L448-L450): restore the best epoch.** The model, and any gender head, get back the weights saved at the best validation UA. Everything below uses them.

**[Lines 451 to 453](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L451-L453): start spike counting.** Only spiking models have `spike_report`; their recorder now counts spikes during the next pass.

**[Lines 454 to 455](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L454-L455): final validation pass.** Validation is predicted again with the restored weights, now with the gender head if one exists.

**[Lines 456 to 461](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L456-L461): spike report.** For spiking models, counting stops and each LIF layer's firing rate on validation is logged, naming any silent layer. The lif\_final run above logged `spikes on validation: lif1 0.185`.

**[Lines 462 to 463](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L462-L463): the test pass.** The held-out test session is predicted once, with the restored weights. Nothing in the job reads the result back to make a choice.

**[Line 464](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L464): final validation metrics.** They come from the second validation pass.

**[Lines 465 to 466](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L465-L466): a consistency check.** If the reloaded weights give a validation UA more than 1e-9 away from the recorded best, the log notes both values.

**[Line 467](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L467): peak GPU memory.** The most memory the job allocated on the GPU, in MiB.

**[Lines 468 to 474](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L468-L474): provenance.** Code hash, package versions, host and environment ([runs/provenance.py lines 60 to 74](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/provenance.py#L60-L74)). Added here: start and finish times, torch, CUDA and cuDNN versions, the GPU's name, the data manifest's SHA-256 and the folds hash.

**[Lines 475 to 479](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L475-L479): metrics\_val.json.** Best epoch, best validation UA, epochs run, whether it stopped early, the curves, and the full validation report at the best epoch. With eval\_train, `train_final` adds the training report of that same epoch, confusion matrix included.

**[Line 480](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L480): metrics\_test.json.** The full test report of the restored weights.

**[Lines 481 to 485](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L481-L485): efficiency.json.** Parameter count, training seconds in total and per epoch (both including the evaluation passes, as at line 446), epochs run, test seconds, milliseconds per test utterance and peak GPU memory.

**[Lines 486 to 493](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L486-L493): optional extras.** Gender-head reports on validation and test, the speed-perturbation cache size, and the spiking models' spike report.

**[Lines 494 to 501](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L494-L501): the returned files.** metrics\_val.json, predictions\_val.csv, metrics\_test.json, predictions\_test.csv, efficiency.json and provenance.json. The two CSV files come from `predictions_csv` (lines 272 to 280).

#### The other blocks of loop.py

**[Lines 1 to 13](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L1-L13): the module note.** One job trains, keeps the epoch with the best validation UA, then predicts validation and test with it. Selection uses validation only, and the status report never reads test predictions. Options added on 2026-09-26 are off in older configurations, which keep the same random draws in the same order.

**[Lines 14 to 31](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L14-L31): imports.** Standard library, numpy, the augmentations, fold assignment, metrics, the feature and model registries, provenance, the job spec and the data helpers.

**[Lines 34 to 37](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L34-L37): `augmentation_rng`.** The separate stream for speed perturbation and mixup, described at line 329.

**[Lines 40 to 46](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L40-L46): `lr_factor`.** The warmup and schedule factor, described at lines 360 to 363.

**[Lines 49 to 82](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L49-L82): `Normalizer`.** The mode comes from features.normalization; "utterance", used by all five records, scales each mel band over the utterance's real frames (lines 68 to 72). Here μ\_b and σ\_b² are band b's mean and variance over those frames:

```latex
\tilde{x}_{b,t} = \frac{x_{b,t} - \mu_b}{\sqrt{\sigma_b^{2} + 10^{-5}}}
```

"global" uses one mean and SD per band from training data, "speaker" one per speaker, and "none" changes nothing. Padded frames are then set to 0 (line 82).

**[Lines 85 to 98](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L85-L98): `global_stats`.** One pass over the given rows as whole utterances, summing each band's values and squares over real frames in float64. It returns each band's mean and SD.

**[Lines 101 to 121](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L101-L121): `speaker_stats`.** The same sums, kept per speaker; it raises an error if a speaker has no frames.

**[Lines 124 to 135](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L124-L135): `make_normalizer`.** Computes what a mode needs: "global" statistics from the training rows, "speaker" statistics from every row's audio, labels unused. "utterance" and "none" need nothing.

**[Lines 138 to 144](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L138-L144): `soft_cross_entropy`.** The emotion and gender loss, described at line 200.

**[Lines 147 to 153](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L147-L153): `make_aux_head`.** Returns None without a gender head. Otherwise it builds dropout plus one linear layer from the model's embedding to two logits ([models/heads.py lines 26 to 43](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/heads.py#L26-L43)).

**[Lines 156 to 157](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L156-L157): `trainable_parameters`.** The model's parameters plus any gender head's, as one list for AdamW.

**[Lines 160 to 165](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L160-L165): `segment_samples`.** Segment length and hop, converted from seconds to samples, or None for whole utterances.

**[Lines 211 to 215](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L211-L215): `Predictions`.** What `predict` returns: class probabilities, gender probabilities or None, and seconds.

**[Lines 272 to 280](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L272-L280): `predictions_csv`.** One row per utterance: utt\_id, speaker, kind, true label, predicted label, then one probability column per class to 6 decimals.

**[Lines 292 to 305](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L292-L305): `gender_report`.** Only with a gender head: accuracy, recall for each gender present, and the mean of those recalls.

### runs/store.py and runs/spec.py: what reaches disk

**[runs/spec.py lines 83 to 84](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py#L83-L84): the run ID.** The first 12 hex digits of the configuration hash, then the fold and the seed, as in 553e54c97626\_test\_S1\_seed6. The hash covers the canonical settings (lines 68 to 80), so a changed setting gives a new name.

**[sweep/worker.py lines 29 to 45](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/worker.py#L29-L45): who writes the files.** The worker opens `artifacts/runs/<sweep_id>/<run_id>.tmp/` and writes spec.json before training starts. It then writes each file `train_job` returned, then log.txt, and only then renames the folder to its final name.

**[runs/store.py lines 33 to 42](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/store.py#L33-L42): opening a folder.** A finished folder is never reopened: opening it raises AlreadyComplete, and the worker skips the job. A .tmp folder left by a crashed job is kept as evidence, renamed with "-abandoned" at the end.

**[runs/store.py lines 52 to 64](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/store.py#L52-L64): writing a file.** Each file is opened in "x" mode, so an existing file is never overwritten. JSON is written with indent 1 and refuses NaN or infinity.

**[runs/store.py lines 66 to 82](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/store.py#L66-L82): complete or fail.** On success the .tmp folder is renamed to the run ID. On failure the traceback goes to error.txt, and the folder becomes `<run_id>.failed-<utc>`.

| File | Code | What it holds |
| --- | --- | --- |
| spec.json | [worker.py line 37](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/worker.py#L37) | The job as queued: settings, fold, seed, config\_hash, run\_id, sweep and stage |
| metrics\_val.json | [loop.py lines 475 to 479](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L475-L479) | Best epoch and UA, epochs run, stopped\_early, per-epoch curves, final validation report, train\_final when measured |
| predictions\_val.csv | [loop.py line 496](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L496) | Per validation utterance: true label, prediction, class probabilities |
| metrics\_test.json | [loop.py line 480](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L480) | The final test report |
| predictions\_test.csv | [loop.py line 498](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L498) | Per test utterance: true label, prediction, class probabilities |
| efficiency.json | [loop.py lines 481 to 493](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L481-L493) | Size, times, peak GPU memory, spike report for spiking models |
| provenance.json | [loop.py lines 468 to 474](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L468-L474) | Code hash, versions, machine, data and fold hashes |
| log.txt | [worker.py line 44](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/worker.py#L44) | Every line the job logged: header, one line per epoch, spikes, notes |
| error.txt | [store.py lines 76 to 78](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/store.py#L76-L78) | Failed runs only: the traceback |

### Questions she may ask about training

- **Why does validation pick the epoch, while test is read only once?** The loop keeps the epoch with the highest validation UA (loop.py lines 433 to 439). It predicts test once, after restoring that epoch (lines 448 and 462). No job of this loop retrains on train plus validation first: test is read from the model trained on three sessions, with its epoch picked on the fourth, and retraining on four sessions is an open decision ([analysis/selection.py lines 10 to 13](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/selection.py#L10-L13)). Picking the epoch on test would inflate the test score; the test\_as\_val protocol exists to measure that inflation ([data/splits.py lines 34 to 37](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L34-L37)). The selection record is written before any test file is read ([analysis/selection.py lines 3 to 8](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/selection.py#L3-L8)).
- **How is training accuracy measured, and does it show over- or underfitting?** With eval\_train "every\_epoch", training utterances are predicted after each validation pass exactly like validation: eval mode, whole utterances, no augmentation, true labels. `curves` then holds train\_ua beside val\_ua for every epoch, and `train_final` holds the full training report of the kept epoch. The mean training loss is not comparable: it includes augmentation, dropout, smoothing, class weights and any rate penalty. cnn\_final\_3 and the baseline predate the field, so the runs behind their records have no training pass. The cnn\_trainacc sweeps re-ran both with every\_epoch on seeds 1 to 5, under session out and pooled; under session out, all 50 of their validation prediction files match, byte for byte, those of the same configuration, fold and seed without the training pass.
- **Why UA, and how is class imbalance handled?** UA is the mean of per-class recalls, so every class counts equally whatever its size. In training, "weighted\_loss" scales each class's loss term by n / (K n\_k), and "balanced\_sampler" draws utterances inversely to class counts. lif\_final, snn\_final and cnn\_final\_3 use weighted\_loss; threeblock\_final and the baseline use neither.
- **What ends training, and how does the schedule interact with early stopping?** A job stops after `epochs` epochs, or once `patience` epochs pass without a strictly higher validation UA (lines 443 to 445). The cosine schedule is laid out over the full `epochs` budget, so a job that stops early never reaches a learning rate of 0. cnn\_final\_3 holds a constant rate after warmup, so this does not apply to it.
- **Where do the LIF penalty and the gender head enter the loss?** Both enter in `Step.loss` (lines 201 to 208), added to the emotion loss. The penalty exists only for spiking models with rate\_weight above 0; lif\_final and snn\_final use rate\_weight 1.0 and rate\_target 0.2. No record here uses a gender head. The kept epoch is still chosen by validation UA alone, never by the loss.
- **Is a run reproducible from its seed?** One seed feeds four generators (lines 326 to 329). They cover initial weights and dropout, epoch order and crops, SpecAugment masks, and speed perturbation with mixup. The seed is also part of the run ID. No cuDNN deterministic flag is set; the comparison module states that repeated runs were identical.
- **Why train on crops but evaluate whole utterances?** Each epoch, training draws one random crop of crop\_s seconds per utterance, zero-padded and masked when shorter. Validation and test see every utterance whole, padded per batch and masked, so no part of an utterance is skipped. The segment option, unused here, would average segment probabilities instead.

## Every other file

Files under models/, features/, augment/ and train/ are covered in other sections.

### Top level

**[README.md](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/README.md)**: the project overview. It covers the layout, the rules the code enforces, the PC and node commands, and the night-by-night history of the sweeps. It also explains `analyze`, `select` and `test-report`; `analyze` needs scipy, so it runs on the node or in the Cowork container, not in the PC's Python.

**[findings.csv](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/findings.csv)**: 162 recorded claims, with columns `finding_id`, `claim`, `runs_and_tables`, `test`, `result` and `status`. IDs N2 to N7 cite the analysis folder of that night; X4 rows cite night 4's `cross_night.json`. Status is "supported" (107 rows) or "not supported (no detectable difference)" (55); nothing in `src/` reads the file.

**[requirements.in](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/requirements.in)**: the training environment's packages, unpinned: torch, numpy, scipy, pandas, matplotlib, soundfile, librosa, papermill, ipykernel, nbformat. `scripts/setup_env.sh` installs them and freezes the exact versions into `artifacts/env/pip_freeze_<utc>.txt`.

**[.gitignore](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/.gitignore)**: keeps out `/data/`, `/artifacts/`, `/queue/`, `/logs/`, `/CODE_VERSIONS.log`, `__pycache__/`, `*.pyc` and `.ipynb_checkpoints/`. It also keeps out `/HANDOFF.md`, `/VERIFICATION_LOG.md` and `/node.local.txt` (the node's login).

**[.gitattributes](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/.gitattributes)**: `*.sh text eol=lf`, because bash on the node rejects shell scripts with Windows line endings.

### src/affect/ (top level)

**[src/affect/\_\_init\_\_.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/__init__.py)**: importing `affect` sets `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` to empty strings ([L25-L26](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/__init__.py#L25-L26)). If torch was imported first without them cleared, it raises ([L18-L23](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/__init__.py#L18-L23)). On the shared node, another user's MPS daemon otherwise makes CUDA unavailable.

**[src/affect/settings.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/settings.py)**: the settings rules. Settings are frozen dataclasses with no defaults. A missing, unknown or mistyped field raises, including a number written as text or a bool for a number. Validated settings are hashed from canonical JSON, and that hash names the run.

- [`check_settings_class`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/settings.py#L92-L103): frozen, no defaults, supported field types only.
- [`validate`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/settings.py#L217-L242) and [`_coerce`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/settings.py#L162-L214): the field-by-field checks. Ints are stored as floats where a float is declared; NaN and infinity are refused; lists become tuples.
- [`added_generations`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/settings.py#L106-L137), [`canonical_plain`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/settings.py#L256-L272), [`settings_hash`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/settings.py#L275-L277): `ADDED_FIELDS` lets a class gain fields. A settings dict must hold every field, or exactly the fields of an earlier date. An added field holding its old value is left out of the hash, so older configurations keep their hashes.

**[src/affect/registry.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/registry.py)**: four registries, `DATASETS`, `FEATURES`, `MODELS` and `CLASSICAL` ([L72-L80](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/registry.py#L72-L80)). A decorator such as `@register_model("cnn", settings=CNNSettings)` names a builder and checks its settings class at import. [`Registry.validate_section`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/registry.py#L51-L69) reads `{"name": ..., fields}` and validates the fields against that class.

**[src/affect/catalog.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/catalog.py)**: [`load_all`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/catalog.py#L7-L14) imports every component once so its registry entry exists. That is dataset "iemocap", feature "logmel", models "cnn", "cnn\_transformer", "cnn\_lif" and "snn", and classical methods "svm" and "tpot".

**[src/affect/cli.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/cli.py)**: the command line, `python -m affect.cli <command>`, with 18 commands listed in its docstring ([L1-L33](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/cli.py#L1-L33)). [`main`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/cli.py#L288-L400) makes every argument required, and each `cmd_*` function imports its module only when called. `verify-data` and `compare-runs` exit 3 when unfinished, meaning "run again".

**[src/affect/env.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/env.py)**: machine and GPU checks. For `env-check`, [`gpu_report`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/env.py#L70-L120) compares a matrix product with the CPU and runs one training step on every GPU. [`require_gpus`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/env.py#L123-L131), which `smoke` and the worker call, refuses to continue unless CUDA sees exactly the expected number of GPUs.

**[src/affect/hashing.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/hashing.py)**: [`sha256_file`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/hashing.py#L7-L18) reads in 1 MiB chunks. [`canonical_json`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/hashing.py#L21-L26) sorts keys, drops whitespace and refuses NaN. [`code_hash`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/hashing.py#L37-L50) hashes every .py file of the package, so two trees hash alike exactly when their Python files match.

**[src/affect/paths.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/paths.py)**: every location is fixed relative to the repository root ([`repo_root`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/paths.py#L10-L15)). That covers `data/<dataset>/raw`, `manifests`, `assembled` and `_work`, `queue/`, and `artifacts/` with `checks`, `sweeps`, `runs`, `analysis`, `selections` and `test_reports`. No code accepts an output path from a caller.

### src/affect/data/

**[src/affect/data/\_\_init\_\_.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/__init__.py)**: package docstring only. A dataset module builds the standard utterance table, declares its label sets and folds, and knows how its assembled copy is built.

**[src/affect/data/iemocap.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py)**: everything IEMOCAP-specific: parsing, label sets, checks of the assembled copy, and the registered dataset. Its docstring ([L1-L20](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L1-L20)) lists the Phase 0 facts it relies on.

- Labels: one summary file per dialog, `SessionN/dialog/EmoEvaluation/<dialog>.txt`. The per-evaluator files in `EmoEvaluation/Attribute/` store valence and activation reversed (6 minus the value); this module never reads them.
- [`parse_label_file`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L104-L178): each header line gives the utterance ID, the released label, valence, activation, dominance and times. "C-E" lines are the external evaluators' categories and "A-E" lines their dimensional scores; self-evaluations are only counted.
- Votes ([L167-L177](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L167-L177)): `votes_all` gives a category one vote from each evaluator who lists it; `votes_first` counts only each evaluator's first category. [`CAT_MAP`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L43-L46) maps names such as "Frustration" to codes such as "fru".
- Speaker ([`UTT_RE`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L40), [L131-L136](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L131-L136)): the session plus the letter in the ID's last field, so `Ses01F_impro01_M000` is `Ses01_M`. The letter after `SesXX` marks who wore the markers. `kind` is "impro" or "script" by the dialog name ("other" otherwise).
- [`LABEL_SETS`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L63-L74) maps raw label to class, and [`CLASS_ORDER`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L75-L84) fixes the output order. An utterance whose raw label is not a key is left out, and "xxx" (no majority) is in no set.

| Label set | Classes, in output order | Utterances (comment at [L57-L62](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L57-L62)) |
| --- | --- | --- |
| `four_merged` | neu, hap+exc, ang, sad | 5,531 |
| `four_without_excited` | neu, hap, ang, sad (excited left out) | 4,490 |
| `five` | neu, hap, exc, ang, sad | 5,531 |
| `six` | neu, hap, exc, ang, sad, fru | 7,380 |
| `seven` | neu, hap, exc, ang, sad, fru, sur | 7,487 |
| `five_with_frustration` | neu, hap+exc, ang, sad, fru | 7,380 |
| `four_with_frustration_no_neutral` | hap+exc, ang, sad, fru | 5,672 |
| `three_no_neutral` | hap+exc, ang, sad | 3,823 |

- The last three are the frustration sets, added 2026-10-03 after Dr. Luitel's meeting of 2026-10-02. Happy and excited stay merged; `three_no_neutral` separates dropping neutral from adding frustration.
- Sweep file names abbreviate these three as `fru5`, `fru4` and `no_neu3`, in that order. For example, `cnn_3block_classes_fru4.json` uses `four_with_frustration_no_neutral`.
- [`IEMOCAPSettings`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L90-L95): `label_set`, `subset` ("all", "impro" or "script", see [`SUBSETS`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L85)) and `split` (one of the 21 schemes below).
- Checks: [`plan_checks`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L456-L497) before copying, [`after_copy_checks`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L500-L543) on hashes, [`check_assembled_copy`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L383-L453) on the result. [`compare_with_reference`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L287-L352) demands field-by-field parity with the Phase 0 index.
- [`IEMOCAP.plan_assembly`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L560-L608): audio comes from the release unless the release file is empty, then from the collaborator's folder. The module docstring counts 501 such files, all in Session 5.
- [`IEMOCAP.utterances`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L619-L643): refuses a copy whose assembly checks failed and keeps the chosen label set and subset. It checks every wav is 16 kHz mono 16-bit PCM and sets `fold_group` to "S1" to "S5".
- [`IEMOCAP.soft_targets`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L645-L664): used when `train.labels` is "soft". It turns each utterance's `votes_all` over the set's classes into a distribution summing to 1, dropping votes outside the set.
- [`IEMOCAP.folds`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap.py#L666-L669): session-out folds directly, every other split through `iemocap_splits.build`.

**[src/affect/data/iemocap\_splits.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap_splits.py)**: the 21 fold schemes behind the data setting `split`. Each has a fixed list of fold names ([`fold_names`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap_splits.py#L102-L119)), which `enqueue` uses to check sweep files without reading data. [`build`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap_splits.py#L334-L377) makes the folds from the utterance table and checks their names.

- Session-out, `leave_one_session_out` (5 folds, `test_S1` to `test_S5`). The groups are the sessions S1 to S5 ([`GROUPS`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap_splits.py#L65)). [`leave_one_out_next_validation`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L64-L75) gives one fold per session.
- In each session-out fold, the next session validates (S5 wraps to S1) and the other three train. So `test_S1` validates on S2 and trains on S3, S4, S5; `test_S5` validates on S1.
- These are plain `Fold` objects of session names. [`assign`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L78-L97) puts each utterance in a role by its session, so both speakers of the test session are tested.
- Pooled, `utterance_5fold` (5 folds, `test_G1` to `test_G5`; [`_utterance_5`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/iemocap_splits.py#L205-L210)). Every utterance is dealt into groups G1 to G5, stratified by session, speaker and class.
- [`next_group_folds`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L160-L173) then tests one group, validates on the next (G5 wraps to G1) and trains on the other three. Speakers and dialogs overlap across roles; the `*_pooled.json` sweep files use this split.
- [`deal`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L138-L157): within each stratum, units are taken in a fixed SHA-256 order ([`stable_order`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L133-L135)) and dealt round-robin, each stratum continuing where the previous one stopped. So every group gets an equal share of each stratum (within one), the groups differ in size by at most one, and the deal is the same on every machine.
- Every other scheme returns [`RoleFold`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L30-L61) objects: explicit sets of utterance IDs per role, which may not overlap, except that `leave_one_session_out_test_as_val` validates on its test set (`selection_on_test`). An utterance in no role is left out of that run.

| Split | Folds | What it holds out |
| --- | --- | --- |
| `leave_one_speaker_out` | 10 | Tests one speaker; the partner from the same session validates; the other 4 sessions train |
| `leave_one_speaker_out_partner_in_train` | 10 | Tests one speaker; a speaker from another session validates; the partner trains |
| `leave_one_speaker_out_enroll_quarter` | 10 | Speaker out with the partner validating, plus a quarter of the test speaker's utterances (dealt by class) moved into training |
| `dialog_5fold` | 5 | Dialogs dealt into 5 groups within each session; speakers overlap, dialogs do not |
| `within_session_5fold` | 25 | Inside each session, utterances dealt into 5 groups by speaker and class |
| `within_session_cross_speaker` | 10 | Inside each session, train on one actor (a fifth validates), test on the partner |
| `within_speaker_5fold` | 50 | Inside each speaker, 5 groups dealt by class |
| `leave_one_session_out_train_female`, `_train_male` | 5 each | Session folds; training and validation keep one gender; the whole session is tested |
| `leave_one_session_out_train_impro`, `_train_script` | 5 each | Session folds; training and validation keep one kind |
| `leave_one_session_out_half_train` | 5 | Session folds; training keeps half, dealt by session, speaker and class |
| `leave_one_session_out_quarter_train`, `_three_quarter_train` | 5 each | Nested training shares: quarter within half within three quarters |
| `leave_one_session_out_random_val` | 5 | Tests one session; a fifth of the other four validates |
| `leave_one_session_out_test_as_val` | 5 | Validation is the test session itself (`selection_on_test=True`): a labeled measurement, never a result |
| `session5_test` | 1 | Sessions 1 to 4 train, a tenth of their dialogs validates, Session 5 tests |
| `leave_one_session_out_script_held_out` | 15 | Tests one script of one session, with that script removed from training and validation |
| `leave_one_session_out_other_script_held_out` | 15 | Same test sets, removing the next script instead: the control |

**[src/affect/data/splits.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py)**: the fold machinery every scheme uses: [`Fold`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L22-L27), [`RoleFold`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L30-L61), [`assign`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L78-L97), [`deal`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L138-L157) and [`next_group_folds`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L160-L173). [`folds_hash`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L117-L128) hashes every fold's role assignment into each run's provenance. `assign` also returns `train_final` (train plus validation), but the training code reads only train, val and test.

**[src/affect/data/ingest.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/ingest.py)**: [`ingest`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/ingest.py#L40-L119) copies a received folder into `data/<dataset>/raw/<name>/` in six resumable phases. It lists, copies with hashing, reads back, re-lists the source and the copy, then writes `manifests/raw_<name>.csv` and `.json`. It stops at `--max-seconds` and continues on the next call.

**[src/affect/data/assemble.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/assemble.py)**: [`assemble`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/assemble.py#L69-L120) builds `data/<dataset>/assembled/` from named sources in six resumable phases. [`_finalize`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/assemble.py#L123-L169) writes `MANIFEST.csv` and `ASSEMBLY.json` only if every check passed. [`verify`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/assemble.py#L172-L262) re-hashes every file against the manifest and repeats the checks (`verify-data`).

**[src/affect/data/transfer.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/transfer.py)**: the copy engine behind ingest and assemble ([`Transfer`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/transfer.py#L90-L210)). A plan is fixed once; each copy logs its SHA-256, then every copy is read back and hashed again. A mismatch, or a source whose size changed, stops the job.

**[src/affect/data/walk.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/walk.py)**: [`ResumableWalk`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/walk.py#L26-L115) lists every file and size under a folder across several time-limited calls. Its docstring gives the reason: listing the 83,151-file release through the PC's shared-folder bridge took about 240 s, longer than one command may run.

**[src/affect/data/manifest.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/manifest.py)**: a manifest is one CSV row per file (`path`, `bytes`, `sha256`, `source`, `source_path`), sorted by path. [`write_manifest`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/manifest.py#L14-L30) never overwrites one. The manifest's own SHA-256 is the data version stamped into every run.

**[src/affect/data/audio.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/audio.py)**: reads wav files with Python's `wave` module, so no audio library is needed. [`read_wav_info`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/audio.py#L26-L34) reads the header; [`read_pcm16_mono`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/audio.py#L37-L50) refuses anything but mono 16-bit PCM at the expected rate.

**[src/affect/data/base.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/base.py)**: [`Utterance`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/base.py#L5-L25) is one row of the standard table, with `fold_group` as the unit a scheme holds out (IEMOCAP: the session). [`Dataset`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/base.py#L28-L67) is the interface every dataset implements: assembly, checks, class order, utterances, folds and fold names.

### src/affect/eval/

**[src/affect/eval/\_\_init\_\_.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/__init__.py)**: package docstring only.

**[src/affect/eval/metrics.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/metrics.py)**: UA (mean per-class recall, the primary metric), WA and macro-F1, all from one [`confusion_matrix`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/metrics.py#L13-L27) (true rows, predicted columns). A class without examples raises. [`report_by_group`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/metrics.py#L83-L101) marks UA undefined in a group that lacks a class.

**[src/affect/eval/stats.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/stats.py)**: the statistics behind `analyze` and the test reports. Every random function takes an explicit numpy Generator, so results repeat exactly.

- [`t_interval`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/stats.py#L21-L34): mean, sample SD (dividing by n minus 1) and a t interval over seeds. The docstring notes that with 5 seeds the 95% half-width is 1.24 times the SD (critical value 2.776).
- [`paired_bootstrap_ua_difference`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/stats.py#L60-L114): the difference in fold-mean UA, model A minus model B, with a percentile interval. Per fold, each model gives utterance IDs, labels, and a seeds-by-utterances table of correct predictions; IDs and labels must match.
- One bootstrap draw ([L96-L108](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/stats.py#L96-L108)): each model's seeds are resampled with replacement, separately. In every fold the utterances are resampled once, and both models share that resample: this is the pairing.
- UA in a draw ([`_fold_ua_per_seed`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/stats.py#L49-L57)) is each class's weighted recall, averaged over classes. A draw that holds no example of some class raises.
- With alpha as 1 minus the confidence, the interval runs from the alpha/2 quantile of the draws to the 1 minus alpha/2 quantile. The two-sided p is twice the smaller share of draws at or below 0, or at or above 0, capped at 1 ([L111](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/stats.py#L111)). With the 2,000 draws `analyze` was run with, the smallest p above 0 is 0.001, so a p of 0 says only that no draw fell past 0. The meeting doc's sweep and grid p-values were therefore recomputed with 1,000,000 draws (artifacts/analyses/weekly\_20261012/doc\_facts\_bootstrap.py).
- [`probability_of_outperforming`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/stats.py#L117-L120): P(A > B) over all pairs of runs, ties counted half. [`probability_of_outperforming_interval`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/stats.py#L123-L135) adds a bootstrap interval.
- Following Bouthillier et al. 2021, the result is "significant" if the interval's lower end exceeds 0.5. It is "meaningful" if the upper end exceeds a threshold (`analyze` passes 0.75).
- [`holm`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/stats.py#L138-L148): Holm's step-down adjustment for m p-values. Sort them, multiply the smallest by m, the next by m minus 1, and so on. Then take a running maximum, cap at 1, and return them in input order.
- Holm example from `tests/test_stats.py`: `holm([0.01, 0.04, 0.03, 0.005])` gives `[0.03, 0.06, 0.06, 0.02]`.

### src/affect/runs/

**[src/affect/runs/\_\_init\_\_.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/__init__.py)**: package docstring only: one job writes one folder, and nothing overwrites anything.

**[src/affect/runs/spec.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py)**: a job's five settings sections: `data`, `features`, `augment`, `model` and `train`. [`validate_job`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py#L43-L55) validates each, and [`check_job`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py#L93-L109) runs cross-section checks without torch. [`config_hash`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py#L79-L80) and [`run_id`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py#L83-L84) name a run `<first 12 hash characters>_<fold>_seed<seed>`.

**[src/affect/runs/store.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/store.py)**: [`OutputFolder`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/store.py#L23-L82) writes into `<id>.tmp/` and renames it to `<id>/` only when complete. Starting a job whose folder is complete raises `AlreadyComplete`; a failure becomes `<id>.failed-<utc>/` with `error.txt`. Files open in exclusive mode, so nothing is overwritten.

**[src/affect/runs/provenance.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/provenance.py)**: [`collect`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/provenance.py#L60-L74) records how a result was made: entry point, arguments, time, code hash, pushed code version, git commit and host. It also records Python and package versions, and the environment variables that change behaviour on the node.

### src/affect/sweep/

**[src/affect/sweep/\_\_init\_\_.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/__init__.py)**: package docstring only (design, the file queue, the worker).

**[src/affect/sweep/design.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py)**: turns a sweep file into jobs, one per configuration, fold and seed. Every configuration is validated and checked before anything is queued, so a typo stops the enqueue instead of failing jobs overnight.

- Format ([L1-L21](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L1-L21)): exactly five top-level keys, `sweep_id`, `description`, `folds`, `base` and `stages` ([`TOP_KEYS`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L30)). Each stage has `name`, `kind`, `priority` (0 to 999, higher runs first) and `seeds`.
- An override names a setting as `"section.field"`; [`apply`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L45-L49) copies the base and [`set_dotted`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L38-L42) refuses a name the base lacks. One level may change several settings together.
- [`levels_of`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L52-L60): a factor named after one setting may list plain values, as shorthand. Any other factor name must list override dicts.
- Stage kinds, inside [`expand`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L78-L224):
  - "base" ([L131-L132](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L131-L132)): the base configuration alone.
  - "variants" ([L133-L135](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L133-L135)): named override sets, such as controls; the label is the variant's name.
  - "grid" ([L136-L149](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L136-L149)): every combination of the factors' levels, except the one equal to the base.
  - "one\_factor\_at\_a\_time" ([L150-L154](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L150-L154)): each level of each factor changed alone from the base. The label is the override, such as `train.lr=0.0003`, after the factor's name for a named factor (`patience: train.early_stopping_patience=20`).
  - "random" ([L156-L189](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L156-L189)): `n_configs` configurations, each taking one level of every factor in `space`. Draws come from Python's `random.Random(sampling_seed)`, so they repeat exactly.
- Random draws: an invalid draw is skipped and counted. A draw that repeats a configuration already in the sweep is skipped without counting. After 50 times `n_configs` attempts it gives up.
- Because repeats are skipped, a second random stage with the same `sampling_seed` continues past the first one's draws. `cnn_transformer_screen_1.json` describes its `random_extra` stage this way.
- Refusals ([L190-L207](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L190-L207)): a non-base configuration identical to the base or to another configuration, and any configuration that changes the data settings. A sweep holds one data protocol, so all comparisons share validation sets.
- Other refusals: folds that are not folds of the base's split ([L96-L103](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L96-L103)), a classical sweep file ([L91-L93](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L91-L93)), and a run ID that appears twice.
- Jobs ([L208-L220](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L208-L220)): for each configuration, every fold, then every seed. Each job carries its stage, label, overrides, priority, config hash, run ID, fold, seed and full settings.

**[src/affect/sweep/enqueue.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/enqueue.py)**: [`enqueue`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/enqueue.py#L29-L59) expands a sweep file and records it in `artifacts/sweeps/<sweep_id>/` (`sweep_file.json`, `summary.json`, `jobs.csv`, `provenance.json`). It then writes one queue file per job. A sweep is enqueued once; a changed sweep needs a new `sweep_id`.

**[src/affect/sweep/queue.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/queue.py)**: a job queue made of files in `queue/pending`, `running`, `done` and `failed`. [`pending_name`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/queue.py#L19-L20) starts each name with 999 minus the priority, so sorted names put the highest priority first. [`Queue.take`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/queue.py#L39-L47) claims a job by an atomic rename, so two workers never take the same job.

- [`stop_requested`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/queue.py#L75-L82): `queue/STOP` stops every worker; `queue/STOP_gpu<g>` stops one GPU's workers.

**[src/affect/sweep/worker.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/worker.py)**: [`run`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/worker.py#L56-L114) takes jobs until the queue is empty, a stop file exists, or free disk falls below `--min-free-gb`. It also stops after `--max-consecutive-failures` failed jobs in a row. [`execute`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/worker.py#L25-L53) runs one job into `artifacts/runs/<sweep_id>/<run_id>/`; every job ends with a line holding a timestamp, then "JOB DONE", "JOB FAILED" or "JOB SKIPPED".

**[src/affect/sweep/status.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/status.py)**: the morning check. [`summarise`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/status.py#L45-L89) ranks each configuration and seed by fold-mean validation UA, using each run's `best_val_ua`. It never reads a test metric. Its alerts ([L74-L83](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/status.py#L74-L83)): baseline below 0.50, a control above 0.30, or any run above 0.80.

**[src/affect/sweep/smoke.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/smoke.py)**: the pre-launch test, on one GPU. [`smoke`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/smoke.py#L155-L226) takes one training step for every distinct configuration and runs short real jobs (2 epochs of 5 batches). The short jobs cover the first job and the first job using each option in [`COVERED`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/smoke.py#L33-L48). It then times the base and prints SMOKE OK or SMOKE FAILED.

### src/affect/analysis/

**[src/affect/analysis/\_\_init\_\_.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/__init__.py)**: package docstring only.

**[src/affect/analysis/validation.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py)**: the `analyze` command: every configuration of a finished sweep against the sweep's baseline, on validation data only. It reads `spec.json`, `metrics_val.json` and `predictions_val.csv` of complete run folders, and never opens a test file.

- [`load`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py#L83-L94) and [`_configs`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py#L97-L107): collect complete runs (skipping `.tmp` and `.failed-` folders) and group them by configuration, seed and fold.
- Score ([`_seed_scores`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py#L110-L116)): per seed, the mean over the sweep's folds of `metrics["final"]["ua"]`. That is validation UA after the best epoch's weights are reloaded. Only seeds with every fold complete count.
- [`_summary`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py#L119-L136), per configuration: mean UA over seeds with a t interval (2 or more seeds), mean WA, macro-F1 and per-class recall. [`_group_scores`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py#L58-L80) adds UA by speaker gender and by improvised or scripted.
- Baseline: the configuration from the stage of kind "base". More than one such configuration raises ([L193-L196](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py#L193-L196)).
- Which are compared ([L199-L205](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py#L199-L205)): every non-control configuration that shares at least two complete seeds with the baseline. Controls are reported apart and never ranked.
- [`_versus`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py#L154-L163), on the shared seeds: per-seed differences with a t interval, and the paired bootstrap on validation predictions ([`_fold_arrays`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py#L139-L151)). It also gives P(configuration beats baseline) with a bootstrap interval (threshold 0.75).
- Verdict ([L206-L213](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py#L206-L213)): Holm's adjustment runs across every compared configuration's bootstrap p. A Holm p below 1 minus the confidence gives "better" or "worse" by the difference's sign; otherwise "no detectable difference".
- [`factor_levels`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py#L166-L186): for each random stage, the mean UA of the configurations drawn with each level of each factor. Factors are sorted by spread, the gap between their best and worst level.
- One generator seeded with `--seed` serves every comparison in turn. The CLI saves `report.json`, `report.txt` and `provenance.json` in `artifacts/analysis/<sweep_id>_<utc>/`.
- [`render`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py#L222-L266) prints a NOTE when reloading the best epoch changed validation UA in some run.

**[src/affect/analysis/groups.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/groups.py)**: [`group_ua`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/groups.py#L30-L36) gives UA per speaker gender, kind or speaker from prediction rows, or None where a group lacks a class. [`route_by_gender`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/groups.py#L39-L57) builds the gender-routed system: the female model answers for female speakers and the male model for male ones.

**[src/affect/analysis/selection.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/selection.py)**: the test stage: record the choice first, then read the test results of that choice only.

- `select` ([`selection_record`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/selection.py#L38-L61)) needs at least 8 characters of the configuration hash, a non-empty reason, and a finished `analyze` report. [`latest_analysis`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/selection.py#L27-L35) takes the newest one.
- The hash prefix must match exactly one configuration, ranked in that report (so not a control, and with a complete seed). The record stores the settings, the reason, the validation summary, and the report's folder and SHA-256.
- The CLI writes `artifacts/selections/<name>/selection.json`. An existing name is refused, so a selection is never overwritten.
- `test-report` ([`test_report`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/selection.py#L69-L110)) reads `metrics_test.json` and `predictions_test.csv` of the chosen configuration's runs only. It reads only seeds with every fold complete and lists the others as incomplete.
- Per seed it takes the fold means of UA, WA, macro-F1 and per-class recall, plus UA by fold, gender and kind. Over seeds it reports the means, with a t interval for UA, WA and macro-F1, saved to `artifacts/test_reports/<name>_<utc>/`.
- Nothing retrains ([L10-L13](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/selection.py#L10-L13)). Test predictions come from the model whose epoch was chosen on the validation session; retraining on 4 sessions is recorded as an open decision.

**[src/affect/analysis/compare.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/compare.py)**: `compare-runs`, which checks that code changes reproduce earlier runs. [`compare_run`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/compare.py#L88-L113) matches each run with every same-named run folder in other sweeps. Validation predictions must match byte for byte and validation metrics exactly (timings removed); test files are compared by SHA-256 only.

- [`compare_sweep`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/compare.py#L116-L143): SAME needs every run complete, identical to every copy, and with a copy made by other code. A differing copy, or a run with no copy made by other code, gives DIFFERENT even while runs are unfinished; otherwise unfinished runs give INCOMPLETE.

### src/affect/classical/

**[src/affect/classical/\_\_init\_\_.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/__init__.py)**: docstring overview of the classical baselines: openSMILE functionals per utterance with an SVM or a TPOT search. They use the same folds, select on validation only, and write the same run-folder files, so `analyze`, `select` and `test-report` read them unchanged.

**[src/affect/classical/settings.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/settings.py)**: four sections: `data`, `functionals`, `method` and `protocol`. [`FunctionalSettings`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/settings.py#L13-L22) picks eGeMAPSv02 (88 features), ComParE\_2016 (6,373) or emobase (988), z-scored globally or per speaker. [`ProtocolSettings`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/settings.py#L25-L31), [`SVMSettings`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/settings.py#L34-L41) and [`TPOTSettings`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/settings.py#L44-L58) hold the rest, with [`check_svm`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/settings.py#L61-L71) and [`check_tpot`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/settings.py#L74-L83).

**[src/affect/classical/functionals.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/functionals.py)**: [`extract`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/functionals.py#L70-L127) runs openSMILE once per feature set over a label set's utterances. It writes `values.npy`, `utt_ids.txt`, `names.txt` and `record.json` into `artifacts/features/<dataset>/<feature_set>/`. [`load`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/functionals.py#L130-L146) returns the rows of a job's utterances and refuses missing ones.

**[src/affect/classical/methods.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/methods.py)**: the two registered methods. [`fit_svm`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/methods.py#L39-L69) tries every C (and every gamma for "rbf") and keeps the best validation UA; scores are the softmax of its decision values. [`fit_tpot`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/methods.py#L95-L124) runs TPOT with one fixed train and validation split, scored by balanced accuracy (UA).

**[src/affect/classical/runner.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/runner.py)**: runs a classical sweep file in one process, without the queue or a GPU. [`expand`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/runner.py#L86-L160) allows stage kinds base, variants and grid. [`run`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/runner.py#L301-L355) goes highest priority first, skips complete runs, and refuses a changed file with an old `sweep_id`.

- [`normalize`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/runner.py#L163-L184) gives z-scores from the training rows or from each speaker's own rows; zero-SD features and missing values become 0. [`run_job`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/runner.py#L228-L269) writes the same files a neural run writes.

**[src/affect/classical/selftest.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/selftest.py)**: [`selftest`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/classical/selftest.py#L36-L84) checks openSMILE on three real utterances (every feature set's width, all values finite). It then needs validation and test UA of at least 0.9 from both SVM kernels on a synthetic four-class problem. A tiny TPOT search must reach test UA of at least 0.5.

### scripts/

**[scripts/common.sh](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/common.sh)**: sourced by the other node scripts. It sets the repository root, the environment `~/venvs/affect`, `HF_HOME`, the empty CUDA MPS variables, 32-thread BLAS caps, and `PYTHONPATH` pointing at `src/`. It also defines `affect_python` and `stamp`.

**[scripts/setup\_env.sh](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/setup_env.sh)**: builds `~/venvs/affect` once from the system Python 3.12 with `requirements.in`, and registers the Jupyter kernel "affect". It writes the exact versions to `artifacts/env/pip_freeze_<utc>.txt` and refuses if the environment already exists.

**[scripts/setup\_classical\_env.sh](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/setup_classical_env.sh)**: builds the separate `~/venvs/classical` from `requirements_classical.txt`, so the training environment never changes. Versions go to `artifacts/env/pip_freeze_classical_<utc>.txt`.

**[scripts/requirements\_classical.txt](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/requirements_classical.txt)**: the classical environment's packages: `setuptools<81`, `numpy==1.26.4`, `scipy==1.13.1`, `scikit-learn==1.4.2`, `pandas==2.2.2`, `xgboost==2.0.3`, `tpot==0.12.2` and `opensmile`. TPOT 0.12.2 is pinned as the last release with the interface `methods.py` uses.

**[scripts/check.sh](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/check.sh)**: the node preflight, stopping at the first failure. It runs every test with nothing allowed to skip (`AFFECT_REQUIRE=numpy,scipy,torch,librosa,data`), then `verify-data`, then `env-check` on 4 GPUs.

**[scripts/push.ps1](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/push.ps1)**: PC side. It packs the code (and with `-Data` the assembled dataset) with SHA-256 files. One `scp` copies them and `unpack.sh` to `~/affect_incoming`, then it prints the node commands. The node's login comes from `node.local.txt`.

**[scripts/unpack.sh](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/unpack.sh)**: node side. It checks the archive's SHA-256, then replaces `src`, `notebooks`, `scripts`, `sweeps`, `tests` and the top-level files wholesale, and logs the install in `CODE_VERSIONS.log`. A data archive goes to `data/<dataset>/assembled/`; installing into `~/affect` is refused while gpu tmux sessions run.

**[scripts/pack.sh](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/pack.sh)**: node side. It packs folders under `artifacts/` into `~/affect_outgoing/<name>.tar.gz` with a list of every file's SHA-256, and refuses while any run folder is unfinished. `--after <earlier archive>` packs only files the earlier archives lacked and stops if one changed.

**[scripts/pull.ps1](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/pull.ps1)**: PC side. It fetches a packed archive, checks its SHA-256, refuses to overwrite anything in `artifacts\`, unpacks it, and checks every file against the node's hash list.

**[scripts/affect.sh](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/affect.sh)**: runs any `affect.cli` command on the node with the project environment, from any folder. Output is copied to `logs/<command>_<utc>.log`.

**[scripts/classical.sh](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/classical.sh)**: the same for the classical commands, with `~/venvs/classical` and one BLAS thread per process. The log name adds the process ID.

**[scripts/worker.sh](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/worker.sh)**: `worker.sh <gpu> <processes>` pins its processes to one GPU with `CUDA_VISIBLE_DEVICES`, 8 BLAS threads each. Each process runs `affect.cli worker` with `--min-free-gb 10` and `--max-consecutive-failures 5`, and logs to `logs/worker_gpu<g>_<n>_<utc>.log`.

**[scripts/smoke\_enqueue.sh](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/smoke_enqueue.sh)**: for each sweep ID given, skips one already enqueued, smoke-tests it (GPU 0, or `SMOKE_GPU`), and enqueues it only after SMOKE OK. It ends with ALL DONE and the number of pending jobs.

**[scripts/gpu\_schedule.sh](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/gpu_schedule.sh)**: the weekend schedule of 2026-10-03 to 10-05, times Pacific. "day" (08:00) writes `STOP_gpu0` and `STOP_gpu1`; "night" (22:00) gives each free GPU 3 workers; "end" writes `queue/STOP`.

- It never starts a GPU another user is on, nor any GPU while `queue/STOP` exists. An idle MPS server does not count as a user; over 2,000 MiB in use does. A "day\_start" step was added on 2026-10-10.

### sweeps/

**[sweeps/](https://github.com/basinhopping/affect/tree/d193f4ed296a20d68477549b91f8fe04af4fc78a/sweeps)**: 180 sweep files, one JSON file per sweep, each named after its `sweep_id` (`tests/test_sweep.py` checks this and expands every file). By base model: 152 `cnn`, 10 `cnn_lif`, 10 `snn`, 5 `cnn_transformer`, and 3 classical (2 `svm`, 1 `tpot`).

**[sweeps/cnn\_3block\_screen\_1.json](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/sweeps/cnn_3block_screen_1.json)**, the worked example. Its `description` quotes Dr. Luitel's request of 2026-10-02: "3 level arch, 32 64 128, focus on a 3 layer, .001 lr, try sweeping around with those settings".

- `folds` ([L4-L10](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/sweeps/cnn_3block_screen_1.json#L4-L10)): `test_S1` to `test_S5`, the five session-out folds. They must be folds of the base's split.
- `base` ([L11-L80](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/sweeps/cnn_3block_screen_1.json#L11-L80)): every setting of the five sections, written out because no setting has a default.
- Base data: `iemocap`, `four_merged`, `all`, `leave_one_session_out`. Features: `logmel` with 64 mel bands. Base model: `cnn` with `channels` \[32, 64, 128\], kernel 3, 2 by 2 pooling, `pooling` "mean", `dropout` 0.3, batch norm, ReLU.
- Base training: `lr` 0.001, `epochs` 60, `batch_size` 32, `early_stopping_patience` 10, `crop_s` 5.0, `precision` "bf16", `eval_train` "every\_epoch". Augmentation is off.
- `stages` ([L81-L321](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/sweeps/cnn_3block_screen_1.json#L81-L321)): three stages, each with its own priority and seeds.
  - `base` ([L82-L93](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/sweeps/cnn_3block_screen_1.json#L82-L93)): kind "base", priority 140, seeds 1 to 5.
  - `one_factor` ([L94-L188](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/sweeps/cnn_3block_screen_1.json#L94-L188)): kind "one\_factor\_at\_a\_time", priority 120, seeds 1 to 3, with a `factors` map. Its levels give 29 configurations.
  - `random` ([L189-L320](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/sweeps/cnn_3block_screen_1.json#L189-L320)): kind "random", priority 100, seeds 1 to 3, `n_configs` 100, `sampling_seed` 20261003, with a `space` map.
- How the one-change stage is written: a key `"section.field"` lists alternative values, each tried alone. Any other key (here `patience` and `specaugment`) lists override dicts, each of which may change several settings together. Every level must differ from the base.
- How the random stage is written: `space` uses the same format. Here every list also includes the base value, and each draw picks one level per factor.
- Trimmed excerpts (the file lists more factors):

```json
{"name": "one_factor", "kind": "one_factor_at_a_time", "priority": 120, "seeds": [1, 2, 3],
 "factors": {"train.lr": [0.0003, 0.0005, 0.002, 0.003],
             "patience": [{"train.early_stopping_patience": 20},
                          {"train.epochs": 100, "train.early_stopping_patience": 20}]}}

{"name": "random", "kind": "random", "priority": 100, "seeds": [1, 2, 3],
 "n_configs": 100, "sampling_seed": 20261003,
 "space": {"train.lr": [0.0003, 0.0005, 0.001, 0.002],
           "model.pooling": ["mean", "stats", "max"]}}
```

- Expanded by `design.expand`: 130 configurations and 1,960 jobs. That is 1 x 5 seeds x 5 folds, plus 29 x 3 x 5, plus 100 x 3 x 5. The notebook prints the same counts.

### notebooks/

**[notebooks/README.md](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/notebooks/README.md)**: what the notebook shows and how to run it: from `notebooks/`, with Python 3.10 or newer, pandas and matplotlib. The settings, check and sweep cells need only the repository; the result cells read `../artifacts/` and say so when it is missing.

**[notebooks/cnn\_3block.ipynb](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/notebooks/cnn_3block.ipynb)**: one experiment from settings to results, in 21 cells. A `SETTINGS` dict holds every setting; `validate_job` and `config_hash` check it (configuration `66d8d74d5623`). A `SWEEP` dict is checked equal to `sweeps/cnn_3block_screen_1.json` apart from its description.

- The result cells plot one run's training and validation UA, then rank the screen and its confirmation (`cnn_3block_confirm_1`) from `analyze` reports. They then show the recorded selections' test results.
- The last cell lists every setting that differs between the base and `threeblock_final`. The saved outputs come from the author's own copy of `artifacts/`.

### tests/

Run with `PYTHONPATH=src python -m unittest discover -s tests -t .`.

- **[tests/\_\_init\_\_.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/__init__.py)**: how to run the tests. Where torch, scipy or the data are missing, tests skip, or fail if `AFFECT_REQUIRE` names them.
- **[tests/support.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/support.py)**: helpers `needs_module` and `needs_assembled`, which skip, or fail under `AFFECT_REQUIRE`.
- **[tests/synthetic\_iemocap.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/synthetic_iemocap.py)**: builds a tiny IEMOCAP-shaped release, collaborator folder and Phase 0 index with a known answer.
- **[tests/test\_analysis.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_analysis.py)**: `analyze` on fake runs with known accuracies gives the expected verdicts, controls, factor levels and saved report, without opening test files.
- **[tests/test\_augment\_data.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_augment_data.py)**: speed perturbation, crops, evaluation segments, gender labels and mixup draws (the numpy parts, without torch).
- **[tests/test\_classical.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_classical.py)**: classical settings, z-scoring, an SVM sweep end to end that `analyze` reads, and TPOT's use of the validation split (needs scikit-learn).
- **[tests/test\_compare.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_compare.py)**: `compare-runs` finds identical copies, ignores timings, catches differences and never parses a test file.
- **[tests/test\_groups.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_groups.py)**: speaker gender parsing, UA by group on a known answer, and the gender-routed predictions.
- **[tests/test\_iemocap\_checks.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_iemocap_checks.py)**: the plan and after-copy checks name each problem on hand-made cases.
- **[tests/test\_iemocap\_data.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_iemocap_data.py)**: the real assembled copy, where present: checks pass, 10,039 utterances, 5,531 in four\_merged, 501 filled files.
- **[tests/test\_iemocap\_end\_to\_end.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_iemocap_end_to_end.py)**: ingest, assemble and verify a synthetic IEMOCAP through the CLI in a throwaway repository copy; a wrong index or changed byte is caught.
- **[tests/test\_iemocap\_parse.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_iemocap_parse.py)**: the label and transcript parsers on hand-written files in IEMOCAP's format.
- **[tests/test\_iemocap\_splits.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_iemocap_splits.py)**: every fold scheme's promises on a synthetic table, plus `deal`, `RoleFold` and `assign`.
- **[tests/test\_import\_guard.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_import_guard.py)**: importing `affect` clears the MPS variables, and refuses when torch came first uncleared.
- **[tests/test\_layout.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_layout.py)**: no .py files outside `src/` and `tests/`, no module-level numeric constants, every settings class frozen without defaults.
- **[tests/test\_lif\_reference.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_lif_reference.py)**: the numpy LIF neuron reference and the spiking models' settings.
- **[tests/test\_lif\_torch.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_lif_torch.py)**: the torch spiking layers against the numpy reference, forward and backward, and both CNN + LIF models (needs torch).
- **[tests/test\_metrics.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_metrics.py)**: UA, WA, F1 and the confusion matrix against hand-computed answers.
- **[tests/test\_registry.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_registry.py)**: registering, duplicate and unknown names, bad settings classes, and that "iemocap" is registered.
- **[tests/test\_selection.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_selection.py)**: `select` refuses before `analyze` and never overwrites; `test-report` then reads the chosen configuration's test files.
- **[tests/test\_settings.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_settings.py)**: missing, unknown and mistyped fields raise; hashing; `ADDED_FIELDS` forms keep their hashes.
- **[tests/test\_splits.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_splits.py)**: session-out folds validate on the next session (S5 on S1), and roles are disjoint and complete.
- **[tests/test\_stats.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_stats.py)**: t interval, Holm, P(A > B), and the paired bootstrap, including a check that pairing narrows the interval.
- **[tests/test\_store.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_store.py)**: output folders complete by rename, are never overwritten, and keep failures as evidence.
- **[tests/test\_sweep.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_sweep.py)**: every sweep file expands, old configurations keep their hashes, bad sweeps are refused, and the queue, enqueue, status and worker loop behave.
- **[tests/test\_torch\_parts.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_torch_parts.py)**: log-mel against librosa, and the models, normalisation, losses, CNN + Transformer, gender head and training step (needs torch).
- **[tests/test\_train\_data.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_train_data.py)**: the torch-free training helpers: speakers, crops, evaluation batches, class balance.
- **[tests/test\_transfer.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_transfer.py)**: the resumable copy finishes under any time budget and catches corruption, truncation and source changes.
- **[tests/test\_walk.py](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_walk.py)**: the resumable listing matches a direct listing and refuses disagreeing rows.

## Questions she may ask about the code

Each model section above ends with its own questions. These cover the code as a whole, then four limits worth raising before she finds them.

### About the code as a whole

- **Where is the 3-block CNN defined?** Not in a file of its own: it is the `cnn` model with three entries in `channels`. One class ([cnn.py lines 213 to 251](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L213-L251)) builds the baseline, cnn\_final\_3 and both 3-block CNNs, and line 222 sets the block count to the length of `channels`.
- **Where does each number come from?** From the settings in the sweep file. Every settings class must be frozen with no defaults ([settings.py lines 92 to 103](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/settings.py#L92-L103)), and [tests/test\_layout.py line 35](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/tests/test_layout.py#L35) fails on any module-level numeric constant in the package. The numbers fixed inside the model code are listed under "How each setting maps to code" in the CNN section.
- **How would a new model be added?** Write a frozen settings class and a builder, decorate the builder with `@register_model("<name>", settings=<class>)`, and add one import line to [catalog.py lines 7 to 14](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/catalog.py#L7-L14). A sweep file whose model section says `"name": "<name>"` then builds it. The registry checks the settings class when the module is imported ([registry.py lines 30 to 41](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/registry.py#L30-L41)).
- **How does one experiment run from start to finish?** As the diagram at the top shows: `smoke` and `enqueue` turn a sweep file into queue files, `worker.sh` runs them into run folders, and `analyze` compares the configurations on validation. `select` then records the choice, and `test-report` reads that choice's test files. The exact commands are in [README.md lines 125 to 141](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/README.md#L125-L141).
- **What keeps test results from steering a choice?** `status` and `analyze` read validation files only. `select` refuses until a finished `analyze` report exists and never reads a test file ([selection.py lines 3 to 8](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/selection.py#L3-L8)). `test-report` then reads the chosen configuration's test files only, and `compare-runs` hashes test files without parsing them.
- **Why do two records hold each baseline?** `cnn_base_session_out` and `cnn_base_session_out_2` hold the same configuration, cf0495a296e9, on seeds 1 to 5 (cnn\_confirm\_1) and 6 to 10 (cnn\_confirm\_2). `threeblock_session_out` and `threeblock_session_out_2` do the same for 66d8d74d5623 (cnn\_3block\_screen\_1 and cnn\_3block\_confirm\_1). The session-out finalists ran on seeds 6 to 10, so they are compared with the `_2` records; rows on seeds 1 to 5 use the first ones.
- **Do the tests pass?** On the node, all 202 passed with nothing skipped on 2026-10-10, just before night 11 ([scripts/check.sh](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/scripts/check.sh)). The edits for GitHub touched no model or training code: comments, one error message, the parse test's made-up fixture, the push and pull scripts, the READMEs and the new notebook. The container ran the same 202 tests on the GitHub copy: OK, with the 40 that need torch skipped, and the node runs all of them again at the next push.

### Limits worth raising first

- **No run retrains on four sessions.** [data/splits.py line 6](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L6) says final runs retrain on train plus validation, but no code does. Every test number comes from the model trained on three sessions, with its epoch picked on the fourth ([selection.py lines 10 to 13](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/selection.py#L10-L13)); whether to retrain is still an open decision, so that line describes a plan, not the code.
- **Padding reaches the convolutions.** Only the pooling over time and the spiking linear layers leave padding out. In training, batch norm's statistics in the CNN blocks and the spiking convolutions include the zeros of crops shorter than `crop_s` ([cnn.py lines 178 to 183](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L178-L183); [snn.py lines 110 to 115](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/snn.py#L110-L115)). At evaluation each batch is padded to its longest utterance, so from the second convolution on, the positions just past a shorter utterance's end hold values computed from its padding, not the zeros it would see alone. The batches are always sorted longest first ([data.py lines 155 to 166](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L155-L166)), so each utterance always gets the same padding.
- **The spiking readout's scale depends on length.** The readout divides by the number of real steps T rather than by its own weights ([lif.py lines 134 to 148](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/lif.py#L134-L148)), so a neuron firing on every step reads below 1 on short sequences. At the spiking finalist's `readout_beta` 0.97 it reads 0.31 over the 25 steps of a 4.0 s training crop, and 0.56 over the 62 steps of a whole 10 s utterance at evaluation; at the hybrid's 0.5 over 8 steps it reads 0.875. Dividing by the sum of the weights, 1 - readout\_beta^(T - t), would make a steady rate read as itself at any length.
- **Determinism rests on seeds and repeat checks, not on a flag.** No cuDNN deterministic flag is set anywhere under src/. The code seeds every generator, and night 2's 160 repeats of night-1 runs were identical ([analysis/compare.py lines 4 to 6](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/compare.py#L4-L6)).
