# Results for the meeting of October 12

Oct 12, 2026 · @Harrison

Of the six CNN models I compare, the tuned 4-block CNN, cnn\_final\_3, scores highest under both splits and every class set. The tuned 3-block CNN and the CNN + LIF hybrid come next: 0.006 and 0.013 UA below it on session out, not a detectable gap, and 0.023 and 0.016 below it pooled, a detectable one. Both beat the baseline detectably in every setting, the fully spiking network is level with the baseline, and the 3-block CNN with the baseline's settings scores lowest.

&#91;embedded content: test files of 12 records · session out, seeds 6 to 10: artifacts/analyses/night10\_20261010/night10\_test.json · pooled, seeds 1 to 5: artifacts/analyses/night11\_20261011/night11\_test.json\]

## Setup

Unless a section says otherwise, every number is unweighted accuracy (UA: the mean of the recall of each class) on IEMOCAP, averaged over 5 training seeds with a 95% interval, and the classes are neutral, happy+excited, angry and sad (5,531 utterances). Differences are computed from the unrounded values, so one can differ by 0.001 from the difference of the two printed numbers.

**Labels.** Each utterance keeps the label IEMOCAP releases. Three external evaluators labeled every utterance, and the released label is the category most of them named, when no other category was named as often: two of the three named it for 5,149 utterances, all three for 2,383. The other 2,507 have no such category; IEMOCAP labels them xxx, and I leave them out. Happy and excited are merged into one class.

| Class | IEMOCAP labels | Utterances |
| --- | --- | --- |
| Neutral | neu | 1,708 |
| Happy+excited | hap (595) and exc (1,041) | 1,636 |
| Angry | ang | 1,103 |
| Sad | sad | 1,084 |
| Frustrated (class sets only) | fru | 1,849 |
| In no class | xxx (2,507), sur (107), fea (40), oth (3), dis (2) | 2,659 |

- **Session out:** train on three sessions, pick the epoch on a fourth, test on the fifth. Test speakers are never seen in training.
- **Pooled:** all five sessions mixed into five folds dealt by speaker and class, so the same speakers appear in training and test.

**The folds.** Both splits have five folds, each with separate training, validation and test utterances. Within a run, validation UA picks the epoch to keep and ends training after a set number of epochs without a new high, the patience ([code](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L433-L448)); across runs, it picks the settings. Each seed's UA is the mean over its five folds. Under session out, each fold tests one session and validates on the next:

| Fold | Test | Validation | Training | Utterances: training / validation / test |
| --- | --- | --- | --- | --- |
| 1 | Session 1 | Session 2 | Sessions 3, 4, 5 | 3,423 / 1,023 / 1,085 |
| 2 | Session 2 | Session 3 | Sessions 1, 4, 5 | 3,357 / 1,151 / 1,023 |
| 3 | Session 3 | Session 4 | Sessions 1, 2, 5 | 3,349 / 1,031 / 1,151 |
| 4 | Session 4 | Session 5 | Sessions 1, 2, 3 | 3,259 / 1,241 / 1,031 |
| 5 | Session 5 | Session 1 | Sessions 2, 3, 4 | 3,205 / 1,085 / 1,241 |

Pooled, I deal the utterances into five groups. Each speaker's utterances of each class are sorted by the SHA-256 hash of the split's name, the speaker and class, and the utterance's ID, then dealt to the groups in turn, each speaker and class continuing where the previous one stopped ([code](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/data/splits.py#L133-L157)). So each group holds a fifth of every speaker's utterances of every class, within one. Fold k tests group k and validates on group k + 1 (group 5 on group 1), which gives 3,318 or 3,319 training, 1,106 or 1,107 validation and 1,106 or 1,107 test utterances. All ten speakers, and every conversation of the test group, also appear in training. The counts, and their check against the runs that used these folds, are in artifacts/analyses/weekly\_20261012/doc\_facts\_data.txt.

**What each result uses.**

| Results | Classes | Split | Seeds |
| --- | --- | --- | --- |
| Lead chart; each "Confirmed on new seeds" | neutral, happy+excited, angry, sad | session out | 6 to 10 |
| Lead chart; each "Pooled and under the class sets" | the same four | pooled | 1 to 5 |
| The class sets; the class sets in each model's section and in Training accuracy | five; four with frustrated for neutral; three | both | 1 to 5 |
| Accuracy per class; Training accuracy; the 3-block CNN as specified | the same four | both | 1 to 5, except the training rows marked 6 to 10 |
| The screens and grids, validation only | the same four | session out | 1 to 3, except: 1 to 5 for the bases of the first CNN, 3-block, LIF, spiking and Transformer screens, for grid 2 and for the augmentation grid; seed 1 alone for the first screen's random draws and for the SVM grid |

A model's test files are read only after its settings were fixed and recorded.

## Accuracy per class

Every CNN struggles most with happy+excited and neutral, and does best on angry and sad. The tuned CNN (cnn\_final\_3) recalls 0.46 of happy+excited and 0.51 of neutral, against 0.68 of angry and 0.75 of sad (session out).

&#91;embedded content: test files of the three rows · artifacts/analyses/night9\_20261009/night9\_confusion.json\]

Pooled, cnn\_final\_3 recalls neutral 0.56, happy+excited 0.56, angry 0.78 and sad 0.81: the same order, higher throughout.

## Training accuracy

The baseline and the 3-block CNN underfit; the tuned CNN, cnn\_final\_3, overfits. At the epoch each run keeps, training UA is 0.66 for the baseline and 0.60 for the 3-block CNN, barely above their validation UA (0.59 and 0.57). cnn\_final\_3 reaches 0.77 against 0.64 (session out).

&#91;embedded content: metrics\_val.json of the 75 runs · artifacts/analyses/night9\_20261009/trainacc\_summary.json\]

| Model | Split | Training UA | Validation UA | Gap | Test UA | Epoch kept / last |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | session out | 0.660 | 0.589 | +0.071 | 0.559 | 16.3 / 26.3 |
| 3-block CNN | session out | 0.597 | 0.570 | +0.027 | 0.537 | 21.2 / 31.2 |
| cnn\_final\_3 | session out | 0.774 | 0.640 | +0.133 | 0.601 | 46.0 / 65.9 |
| Baseline | pooled | 0.647 | 0.608 | +0.038 | 0.602 | 13.1 / 23.1 |
| 3-block CNN | pooled | 0.603 | 0.592 | +0.011 | 0.585 | 24.5 / 33.8 |
| cnn\_final\_3 | pooled | 0.886 | 0.687 | +0.200 | 0.678 | 72.1 / 89.2 |
| 3-block CNN after the sweep | session out, seeds 6 to 10 | 0.711 | 0.634 | +0.077 | 0.601 | 62.8 / 82.4 |
| CNN + LIF hybrid | session out, seeds 6 to 10 | 0.775 | 0.626 | +0.149 | 0.595 | 42.4 / 57.4 |
| Fully spiking CNN + LIF | session out, seeds 6 to 10 | 0.690 | 0.584 | +0.106 | 0.555 | 41.3 / 57.1 |

Values at the epoch each run keeps (its best validation epoch), means over the 25 runs; test UA of the same runs. The first six rows come from artifacts/analyses/night9\_20261009/trainacc\_summary.json, the last three from artifacts/analyses/night10\_20261010/night10\_trainacc.json.

The baseline and the 3-block CNN stop early: their validation UA swings from epoch to epoch, and early stopping (patience 10) ends them while training UA is still near 0.6. cnn\_final\_3 (learning rate 0.0003, patience 20, up to 100 epochs, SpecAugment) keeps fitting the training set, past 0.8 around epoch 60, while validation stays near 0.58. The class sets show the same split: the gap is 0.10 to 0.21 for cnn\_final\_3 and at most 0.06 for the other two.

After the sweep, the 3-block CNN no longer stops early: it fits its training set further while its validation UA rises. Both LIF models fit the training set well beyond validation, the hybrid most. Pooled and under the class sets, the hybrid still has the largest gap between training and validation UA in every setting, 0.095 to 0.160; the fully spiking network's is 0.065 to 0.102 and the tuned 3-block CNN's 0.038 to 0.079.

## The class sets

Frustration is the hardest class to add. With four classes either way, swapping neutral for frustration lowers cnn\_final\_3's UA from 0.601 to 0.580 (session out). Five classes reach 0.505 against a chance level of 0.20, and three classes (no neutral, no frustration) 0.723 against 0.33.

| Classes | Utterances | Chance | cnn\_final\_3, session out | cnn\_final\_3, pooled | Baseline, session out | Baseline, pooled |
| --- | --- | --- | --- | --- | --- | --- |
| Neutral, happy+excited, angry, sad | 5,531 | 0.25 | 0.601 | 0.678 | 0.559 | 0.602 |
| Five: those four and frustrated | 7,380 | 0.20 | 0.505 | 0.570 | 0.439 | 0.495 |
| Four: frustrated in place of neutral | 5,672 | 0.25 | 0.580 | 0.641 | 0.521 | 0.572 |
| Three: happy+excited, angry, sad | 3,823 | 0.33 | 0.723 | 0.783 | 0.683 | 0.735 |

Under session out, every set uses the same five folds with its own utterances: each fold tests one whole session, so the test sets hold:

| Set | Session 1 | Session 2 | Session 3 | Session 4 | Session 5 |
| --- | --- | --- | --- | --- | --- |
| Neutral, happy+excited, angry, sad | 1,085 | 1,023 | 1,151 | 1,031 | 1,241 |
| Five: those four and frustrated | 1,365 | 1,348 | 1,533 | 1,512 | 1,622 |
| Four: frustrated in place of neutral | 981 | 986 | 1,213 | 1,254 | 1,238 |
| Three: happy+excited, angry, sad | 701 | 661 | 831 | 773 | 857 |

Pooled, each set is dealt into five groups of its own, as in Setup: 1,476 utterances per group for five classes, 1,134 or 1,135 for four with frustrated, and 764 or 765 for three.

Under each of the three new sets and both splits, cnn\_final\_3 beats the baseline by 0.040 to 0.075 (Holm-adjusted p 0.006 for each). The 3-block CNN is no better than the baseline under any set.

With the four other models added, cnn\_final\_3 stays first and the 3-block CNN with the baseline's settings last under every set and split. Only the three-class set takes any of them past 0.70 UA.

&#91;embedded content: test files of 36 records, seeds 1 to 5 · artifacts/analyses/night11\_20261011/night11\_test.json\]

&#91;embedded content: test files of tuned3\_classes\_fru5\_final\_3 and tuned3\_classes\_fru4\_final\_3 · night9\_confusion.json\]

Frustration has the most utterances (1,849) but sits between the other classes: its misses spread almost evenly, and 0.18 to 0.20 of angry utterances are labelled frustrated. Without neutral, happy+excited is recognized more often: 0.55 with frustration in neutral's place and 0.59 among three classes, against 0.46 among the standard four.

## The 3-block CNN and the sweep around it

With the baseline's other settings, the 3-block CNN (32, 64, 128 channels, learning rate 0.001) scores 0.537 on session out and 0.585 pooled. That is 0.022 and 0.016 below the 4-block baseline, and 0.064 and 0.093 below cnn\_final\_3 (Holm-adjusted p 0.003 or lower for all four). It underfits: see Training accuracy.

**The sweep.** Around it I ran 129 configurations on validation, each on 3 seeds: 29 that change one setting and 100 random combinations. The best reaches 0.630 validation UA against the base's 0.568 on the same seeds. How I searched the settings, below, defines every setting it changed.

- One change alone: only a 5×5 kernel helps detectably (+0.025). Training longer (patience 20) adds about 0.02, not detectable after correcting for 129 comparisons.
- The five best all use 5×5 kernels, SD pooling and up to 100 epochs with patience 20; four of them also use wider layers (64, 128, 256).

**Pooling and SD pooling.** Pooling happens in two places. Inside each block, a 2×2 max pooling halves the frequency and time resolution. After the last block, the remaining time steps (of a random crop in training, of the whole utterance in validation and test) are summarized into one vector: mean pooling averages each channel over time, and SD pooling (statistics pooling) adds each channel's standard deviation over time to its mean, so the classifier also sees how much each feature varies.

Exactly: after the last block the remaining mel bands are averaged, leaving one value h for each channel c at each time step t. The steps of padding added to shorter utterances are left out (m\_t is 0 on them and 1 on real steps), and n is the number of real steps:

```latex
\begin{aligned}
\text{mean:}\quad & \mu_c = \frac{1}{n}\sum_{t} m_t\, h_{c,t} \\
\text{SD (statistics):}\quad & \sigma_c = \sqrt{\frac{1}{n}\sum_{t} m_t\,(h_{c,t} - \mu_c)^2 + 10^{-6}}, \qquad \text{output } [\mu_1, \dots, \mu_C,\ \sigma_1, \dots, \sigma_C] \\
\text{max:}\quad & M_c = \max_{t:\ m_t = 1} h_{c,t} \\
\text{attention:}\quad & \mathbf{e} = \sum_{t} \alpha_t\, \mathbf{h}_t, \qquad \alpha_t = \frac{m_t\, e^{s_t}}{\sum_{\tau} m_{\tau}\, e^{s_{\tau}}}, \qquad s_t = \mathbf{w}^{\top} \mathbf{h}_t + b
\end{aligned}
```

SD pooling divides by n, so its variance is the population variance, and the 1e-6 keeps the square root and its gradient finite when a channel does not change. It doubles the classifier's input: 512 values for the tuned 3-block CNN's 256 channels. In attention pooling, w and b are learned with the rest of the network ([code](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L98-L121)).

&#91;embedded content: the CNN's layers, from src/affect/models/cnn.py\]

Swapped in alone, SD pooling changed validation UA by +0.007, not a detectable difference; max pooling changed it by +0.001 and attention pooling by -0.016, neither detectably. Among the 100 random configurations, those with SD pooling averaged 0.595, against 0.584 with mean pooling and 0.569 with max pooling. On the 4-block baseline, the first grid search found SD pooling slightly worse (see How I searched the settings, below).

**Tested directly on both finalists.** Last night I ran each finalist again with only its pooling changed, on seeds 6 to 10, and fixed which comparisons I would read before any result existed. Test UA, session out:

| Pooling | Tuned 3-block CNN | cnn\_final\_3 |
| --- | --- | --- |
| SD | 0.601 (its own) | 0.608 |
| Mean | 0.576 | 0.607 (its own) |
| Max | 0.594 | 0.600 |
| Attention | 0.564 | 0.583 |

In the tuned 3-block CNN, SD pooling beats mean pooling by 0.025 and attention pooling by 0.037 (Holm-adjusted p 0.0015 for both); max pooling is 0.007 lower, not a detectable difference. In cnn\_final\_3, SD and mean pooling are level (+0.001, Holm-adjusted p 0.777), max pooling is 0.007 lower (not detectable) and attention pooling 0.024 lower (Holm-adjusted p 0.0015). So SD pooling helps the 3-block CNN and makes no detectable difference in cnn\_final\_3, and attention pooling is worse in both. The numbers are in artifacts/analyses/night12\_20261012/night12\_test.json.

**Confirmed on new seeds.** I re-ran the base and the 19 best configurations on seeds 6 to 10, which the sweep never used, and tested only the best on validation. On test it beats the 3-block CNN with the baseline's settings by 0.066 and the baseline by 0.048 (Holm-adjusted p 0.0015 for both). It is 0.006 below cnn\_final\_3, not a detectable difference (p 0.245). The base again scored below the baseline on these seeds (-0.019).

| Setting | 3-block CNN, baseline's settings | Best after the sweep |
| --- | --- | --- |
| Channels | 32, 64, 128 | 64, 128, 256 |
| Kernel | 3×3 | 5×5 |
| Pooling after the last block | mean | SD |
| Hidden layer before the output | none | 128 units |
| Dropout | 0.3 | 0.2 |
| Batch size | 32 | 16 |
| Training crop | 5 s | 10 s |
| Mel bands | 64 | 40 |
| SpecAugment | none | 2 frequency masks of up to 12 bands, 2 time masks of up to 30 frames |
| Epochs (patience) | up to 60 (10) | up to 100 (20) |
| Learning rate | 0.001 | 0.001 |

**Pooled and under the class sets.** I trained this configuration again with only the data changed, on seeds 1 to 5. Pooled, it scores 0.654: 0.069 above the 3-block CNN with the baseline's settings, 0.053 above the baseline and 0.023 below cnn\_final\_3 (Holm-adjusted p 0.0015 for all three). Under each class set and split it beats the baseline by 0.029 to 0.048 (Holm-adjusted p 0.009 for each) and sits 0.011 to 0.030 below cnn\_final\_3, a gap I did not test.

## How I searched the settings

I searched in two ways. A grid tries every combination of a few settings, so it shows how they interact. A screen tries random combinations of many settings around a base; four of the screens (the first CNN screen, the 3-block sweep and the two LIF screens) first change one setting at a time. I used grids where I wanted the interactions among a few settings (four in the two CNN grids, three in the augmentation and SVM grids) and screens everywhere else, because a grid over just the values the 3-block sweep drew its random combinations from would need 11,943,936 configurations.

**The two grids of the CNN's settings.** Both start from the 4-block baseline and use four classes, session out and validation only.

|  | Grid 1 (night 2) | Grid 2 (night 3) |
| --- | --- | --- |
| Why | Requested on 2026-09-25: the four settings with the clearest effects in the first screen, crossed to see how they interact | Four settings with large gaps between their values' mean UA in the second screen (the four largest apart from SpecAugment in its 200 extra random draws), crossed around the baseline with mild SpecAugment, grid 1's largest average effect |
| Base | The baseline: 64 mel bands, no SpecAugment, no class weighting, mean pooling | The baseline with mild SpecAugment |
| Settings crossed | Mel bands 64 or 40; SpecAugment off or mild (2 masks of up to 8 bands, 2 of up to 20 frames); class weighting none or weighted loss; pooling mean or SD | Patience 10 or 20; 4 or 5 blocks (32 to 256, or 32 to 512 channels); normalization per utterance or per speaker; training crop 5 or 7.5 s |
| Runs | 16 combinations × 3 seeds × 5 folds = 240 | 16 combinations × 5 seeds × 5 folds = 400 |
| Best combination | 40 mel bands with mild SpecAugment: 0.609 against the base's 0.588, +0.021 (Holm-adjusted p 0.024), the only detectable one of 15; +0.027 when re-run on seeds 6 to 10 | Patience 20 with 5 blocks: 0.618 against 0.596, +0.022 (Holm-adjusted p 0.0005); 6 of 15 detectable, all with patience 20 |
| Average effect of each setting (Holm-adjusted p over 10 terms) | Mild SpecAugment +0.010 (0.006); 40 mel bands +0.007 (0.098); class weighting +0.001 (1.0); SD pooling -0.008 (0.043) | Patience 20 +0.015 (below 0.001); 5 blocks +0.008 (below 0.001); per-speaker normalization -0.003 (0.101); 7.5 s crop -0.004 (0.044) |

An average effect is the mean over the 8 combinations with a setting minus the mean over the 8 without it, from a least-squares fit with the seeds as blocks. Per-speaker normalization takes its statistics from each test speaker's own unlabeled audio. In grid 2, patience 20 helped less with per-speaker normalization (+0.011 against +0.018, Holm-adjusted p 0.039). Two more grids, of three settings each, covered augmentation (speed perturbation, mixup and evaluating in segments, on cnn\_final\_1: 18 combinations on 5 seeds) and the SVM baseline (feature set, z-scoring and kernel: 12 combinations on one seed). The numbers for grids 1 and 2 and the 3-block sweep are in artifacts/analyses/weekly\_20261012/doc\_facts\_search.txt; the other two grids are defined in sweeps/cnn\_augment\_1.json and sweeps/classical\_svm\_1.json. The bootstrap p-values in this section use 1,000,000 draws (doc\_facts\_bootstrap.txt there), because the stored analyses' 2,000 draws cannot resolve a p below 0.001.

**Pooling in the grid.** On the 4-block baseline, SD pooling lowered validation UA by 0.008 on average across the grid. It cost 0.013 without SpecAugment and 0.003 with it, though that difference is not detectable after the correction (Holm-adjusted p 0.341). In the 3-block sweep's random draws the order reversed: configurations with SD pooling averaged 0.595, against 0.584 with mean pooling, a gap I did not test. So SD pooling's value may depend on the rest of the configuration; a grid can measure that and a one-change test cannot; the direct test on both finalists, in the 3-block section, found SD pooling detectably better than mean pooling in the tuned 3-block CNN and level with it in cnn\_final\_3.

**Why not one grid over everything.** The 3-block sweep's random combinations drew 16 settings from 2 to 4 values each. Every combination of those is 11,943,936 configurations, or 179,159,040 runs on 3 seeds and 5 folds; the whole sweep ran 1,960. A grid small enough to run would have to drop most settings or values in advance. Random combinations instead let each setting's values appear many times within the same budget (17 to 55 configurations per value in the 3-block sweep). That suits the usual case: for most data sets only a few settings matter, and different ones matter on different data sets ([Bergstra and Bengio 2012](https://jmlr.org/papers/v13/bergstra12a.html)). So the screens draw random combinations, and the best are confirmed on more seeds before any test file is read: from the second confirmation on, on seeds 6 to 10, which no screen used. The first confirmation, which chose cnn\_final\_1, ran on seeds 1 to 5, which include seeds the first screen had used.

**Every setting the 3-block sweep changed.** The base is the 3-block CNN with the baseline's other settings.

| Setting | What it changes | Base | Alone: change in validation UA | Random draws: best and worst level |
| --- | --- | --- | --- | --- |
| Channels | Feature maps in each of the 3 blocks: the network's width | 32, 64, 128 | 16, 32, 64: -0.009; 48, 96, 192: -0.007; 64, 128, 256: -0.003 | 64, 128, 256: 0.592; 16, 32, 64: 0.573 |
| Kernel | Size of each square convolution filter over frequency and time: mel bands by frames in the first block, pooled rows by steps after that | 3×3 | 5×5: **+0.025** | 5×5: 0.593; 3×3: 0.574 |
| Convolutions per block | Convolution, batch norm and ReLU layers before each block's max pooling | 1 | 2: +0.015 | 1: 0.586; 2: 0.581 |
| Pooling after the last block | How the time steps become one vector, as defined above | mean | SD: +0.007; max: +0.001; attention: -0.016 | SD: 0.595; max: 0.569 |
| Hidden layer | An extra layer of 128 units, with ReLU, before the output | none | 128 units: +0.005 | none: 0.590; 128 units: 0.578 |
| Dropout | Share of the pooled values, and of the hidden layer's outputs when there is one, set to 0 at random in training | 0.3 | 0.1: +0.001; 0.5: -0.013 | 0.2: 0.589; 0.5: 0.574 |
| Learning rate | Peak step size of the AdamW optimizer, reached after 2 warm-up epochs | 0.001 | 0.0003: -0.005; 0.0005: 0.000; 0.002: +0.009; 0.003: +0.012 | 0.001: 0.589; 0.002: 0.578 |
| Schedule | The learning rate after warm-up: a cosine decay to 0 at the last allowed epoch, or constant | cosine | constant: -0.004 | cosine: 0.587; constant: 0.578 |
| Training length | The most epochs, and how many epochs without a higher validation UA end training (patience) | 60 epochs, patience 10 | 60, patience 20: +0.022; 100, patience 20: +0.018 | 100, patience 20: 0.592; 60, patience 10: 0.568 |
| Batch size | Utterances per optimizer step | 32 | 16: +0.012; 64: -0.001 | 16: 0.587; 64: 0.577 |
| Weight decay | AdamW's pull of every weight toward 0 at each step | 0.0001 | 0.001: +0.005; 0.01: +0.003 | 0.003: 0.589; 0.01: 0.581 |
| Training crop | Seconds cut at random from each training utterance each epoch; shorter ones are padded and the padding is masked | 5 s | 7.5 s: 0.000; 10 s: +0.006 | 7.5 s: 0.589; 5 s: 0.579 |
| Class balance | Weights each class's loss term by n / (K n\_k), so rarer classes count more | none | weighted: +0.008 | none: 0.585; weighted: 0.582 |
| Labels | Hard: one class per utterance. Soft: each class's share of the evaluators' votes, counting only votes for the four classes (an evaluator who named several categories votes for each) | hard | soft: +0.006 | hard: 0.586; soft: 0.579 |
| Mel bands | Rows of the log-mel spectrogram, its frequency resolution | 64 | 40: +0.010; 80: +0.004 | 40: 0.591; 64: 0.579 |
| SpecAugment | Random masks on the training log-mels: stripes of mel bands and of frames set to 0 | none | 2 masks of up to 8 bands and 2 of up to 20 frames: -0.002 | none and that one: 0.584; 2 of up to 12 bands and 2 of up to 30 frames: 0.581 |

Alone: each of the 29 single changes swapped into the base by itself, on seeds 1 to 3, against the base's 0.568 on the same seeds. Only the 5×5 kernel is detectable after correcting for the 129 comparisons (Holm-adjusted p 0.002); patience 20's +0.022 has Holm-adjusted p 0.108, and attention pooling's -0.016 has 0.963. Of the 100 random combinations, 34 are detectably better than the base and 6 worse. Random draws: the mean validation UA of the configurations drawn with that level, 17 to 55 of the 100 per level. Dropout 0.2 and 0.4, weight decay 0.003 and the stronger SpecAugment appear only in the random draws; learning rate 0.003, dropout 0.1 and attention pooling only alone. In the class balance formula, n is the number of training utterances, K the number of classes and n\_k the count of class k.

## CNN + LIF

I tried leaky integrate-and-fire (LIF) neurons in two designs, both tuned on session out with the four classes:

- **Hybrid:** the 3-block CNN's front end (32, 64, 128 channels), then one to three layers of LIF neurons and a leaky-integrator readout.
- **Fully spiking:** LIF neurons after every convolution, spikes max-pooled, then the same readout.

On validation, the hybrid starts at 0.586 and the best of its 132 variants reaches 0.626. The fully spiking network starts at 0.550, and the best of its 95 variants reaches 0.581.

**Confirmed on new seeds.** Each design's base and its 9 best configurations ran again on seeds 6 to 10, and only the best on validation was tested. The hybrid beats the baseline by 0.041 (Holm-adjusted p 0.002) and sits 0.013 below cnn\_final\_3 (p 0.036 before correcting for the four comparisons, 0.072 after). Its best configuration has four convolution blocks (32 to 256 channels) and one layer of 256 LIF neurons; a penalty pulls each neuron's firing rate toward 0.2. The fully spiking network is level with the baseline (+0.001, p 0.93) and 0.052 below cnn\_final\_3 (Holm-adjusted p 0.002).

**Pooled and under the class sets.** I trained both best configurations again with only the data changed, on seeds 1 to 5.

- **Hybrid:** pooled, 0.662, which is 0.060 above the baseline (Holm-adjusted p 0.002) and 0.016 below cnn\_final\_3 (Holm-adjusted p 0.006). Under each class set and split it beats the baseline by 0.029 to 0.049 (Holm-adjusted p 0.009 for each).
- **Fully spiking:** pooled, 0.615, which is 0.013 above the baseline, not detectable (Holm-adjusted p 0.09), and 0.063 below cnn\_final\_3 (Holm-adjusted p 0.002). Under the class sets it differs from the baseline by -0.001 to +0.016, never detectably.

## Published CNN accuracy on IEMOCAP

CNNs above 70% exist, but every one I could check uses a protocol that raises the number: speakers shared between training and test, only the improvised half of the data, or an epoch or model picked on the test data. Under clean speaker-independent protocols, published from-scratch CNNs reach 56.6 to 61.8 UA. cnn\_final\_3 reaches 0.601 to 0.607 while training on three sessions instead of four.

| Paper | Model | Data and split | Reported | What raises it |
| --- | --- | --- | --- | --- |
| [Zhao et al. 2019](https://doi.org/10.1016/j.bspc.2018.08.035) | 2D CNN + LSTM | same speakers in training and test | 89.2% accuracy | shared speakers; 52% speaker independent ([cited here](https://arxiv.org/pdf/2403.14083)) |
| [Li et al. 2019](https://www.isca-archive.org/interspeech_2019/li19n_interspeech.pdf) | CNN + BLSTM + attention, gender task | all 5,531, session out | 82.8 UA | best of 100 epochs kept, no validation set; 72.6 without the gender task |
| [Mustaqeem and Kwon 2020](https://www.mdpi.com/1424-8220/20/1/183/html) | CNN | four classes, random 80/20 split of utterances | 81.75% | shared speakers |
| [Xu et al. 2021](https://arxiv.org/pdf/2102.01813) | CNN + area attention | improvised only, random 80/20 | 77.5 UA | improvised only, shared speakers, model kept on its test score |
| [Aftab et al. 2022 (Light-SERNet)](https://arxiv.org/pdf/2110.03435) | fully convolutional CNN | all 5,531, 10 folds | 70.8 UA | folds split by session and marker-wearing actor, so test speakers are also in training ([code](https://raw.githubusercontent.com/AryaAftab/LIGHT-SERNET/master/filter_dataset.py)); test fold used as validation ([code](https://raw.githubusercontent.com/AryaAftab/LIGHT-SERNET/master/train.py)) |
| [Aldeneh and Mower Provost 2017](https://emp.engin.umich.edu/wp-content/uploads/sites/67/2022/06/2017_Aldeneh_ICASSP.pdf) | CNN, speed perturbation | all 5,531, speaker out, partner validates | 61.8 UA | clean |
| [Satt et al. 2017](https://www.isca-archive.org/interspeech_2017/satt17_interspeech.pdf) | CNN; CNN + LSTM | improvised only, session out, one speaker validates | 56.6; 59.4 UA | clean |
| This project: cnn\_final\_3 | CNN | all 5,531, session out, next session validates | 0.601 to 0.607 UA | clean |
| This project: cnn\_final\_3, pooled | CNN | all 5,531, pooled folds | 0.678 UA | shared speakers |

The same model loses a lot when speakers are kept apart: Zhao et al. report 89% with shared speakers and 52% without, and my own CNN drops from 0.678 pooled to 0.601 under session out. Fewer classes raise it without any leakage: among three classes (happy+excited, angry, sad), cnn\_final\_3 reaches 0.723 on session out; pooled, with shared speakers too, 0.783.

## Other models, and why the CNN sits where it does

Published results with pretrained speech models are the highest on this data. Under session out, frozen pretrained models with a small classifier reach 63.4 to 72.9% accuracy, against 0.58 for cnn\_final\_3 on the same measure. The three models in the table trained from scratch land within 0.02 UA of each other.

| Model | Training | Size | UA | Accuracy | Source |
| --- | --- | --- | --- | --- | --- |
| SVM on openSMILE ComParE features, per-speaker normalization | from scratch |  | 0.621 | 0.586 | this project, one run |
| cnn\_final\_3 | from scratch | 1.1M | 0.607 | 0.579 | this project, seeds 6 to 10 |
| CNN + Transformer (cnn\_transformer\_final\_1) | from scratch | 0.95M | 0.605 | 0.580 | this project, seeds 6 to 10 |
| wav2vec 2.0 Base (LibriSpeech, 960 h) | frozen, weighted sum of layers | 95M |  | 63.4% | [SUPERB](https://arxiv.org/abs/2105.01051) |
| HuBERT Base (LibriSpeech, 960 h) | frozen, weighted sum of layers | 95M |  | 64.9% | [SUPERB](https://arxiv.org/abs/2105.01051) |
| wav2vec 2.0 Large (Libri-Light, 60k h) | frozen, weighted sum of layers | 317M |  | 65.6% | [SUPERB](https://arxiv.org/abs/2105.01051) |
| data2vec Base (LibriSpeech, 960 h) | frozen | 94M |  | 67.4% | [emotion2vec](https://arxiv.org/abs/2312.15185) |
| HuBERT Large (Libri-Light, 60k h) | frozen, weighted sum of layers | 317M |  | 67.6% | [SUPERB](https://arxiv.org/abs/2105.01051) |
| WavLM Base+ (94k h) | frozen, weighted sum of layers | 95M |  | 68.7% | [WavLM](https://arxiv.org/abs/2110.13900) |
| WavLM Large (94k h) | frozen, weighted sum of layers | 317M |  | 70.6% | [WavLM](https://arxiv.org/abs/2110.13900) |
| UniSpeech-SAT Large (94k h) | frozen, weighted sum of layers | 317M |  | 70.7% | [UniSpeech-SAT](https://arxiv.org/abs/2110.05752) |
| emotion2vec Base (LibriSpeech 960 h plus 262 h of emotional speech, IEMOCAP included) | frozen | 94M | 72.7% | 71.8% | [emotion2vec](https://arxiv.org/abs/2312.15185) |
| Whisper large-v3 encoder (5M h of weakly and pseudo-labeled audio) | frozen, last-layer features | 1,550M whole model | 73.5% | 72.9% | [EmoBox](https://arxiv.org/abs/2406.07162) |
| wav2vec 2.0 Base (LibriSpeech, 960 h) | frozen, all layers, per-speaker normalization | 95M | 67.2% |  | [Pepino et al. 2021](https://arxiv.org/abs/2104.03502) |
| wav2vec 2.0 Base (LibriSpeech, 960 h) | fine-tuned 15 epochs, no validation | 95M | 69.9%; 74.3% after more pretraining on IEMOCAP |  | [Chen and Rudnicky 2023](https://arxiv.org/abs/2110.06309) |
| HuBERT, size not stated | fine-tuned |  | 73.5%; 75.8% with checkpoint averaging | 73.0%; 75.2% | [Morais et al. 2022](https://arxiv.org/abs/2202.03896) |
| HuBERT Large (Libri-Light, 60k h) | partly fine-tuned, leave two speakers out | 317M |  | 73.0% | [Wang et al. 2021](https://arxiv.org/abs/2111.02735) |

**Reading the table.** Every row uses four classes with excited merged into happy, like my four-class set, and session out unless its Training cell says otherwise. Each paper reports accuracy, UA or both, and I copied only what it reported. Size is the number of parameters (mine from each run's efficiency.json). Frozen means only a small classifier is trained on the pretrained model's features; in the SUPERB, WavLM, UniSpeech-SAT and Pepino et al. rows it reads a learned weighted average of all the model's layers. Fine-tuned means the pretrained model is trained as well.

Where the rows need care:

- emotion2vec's pretraining audio includes IEMOCAP, so its test speakers were heard, without labels, in pretraining.
- EmoBox chose its settings by the best outcome and does not say on which split.
- Pepino et al.'s 67.2% normalizes each speaker with statistics from all of that speaker's audio, test speakers included; with training statistics only it is 65.8%.
- My SVM's per-speaker z-scores also use each test speaker's unlabeled audio, as Pepino et al.'s normalization does; the two CNNs in the table normalize each utterance on its own.
- Chen and Rudnicky trained a fixed 15 epochs without a validation set, and their 74.3% first continues pretraining on IEMOCAP's audio; the paper does not say whether the test session was left out of it.
- Wang et al. fine-tune all but the convolutional front end and test one speaker per fold, with another for validation; with random splits that share speakers the same model reaches 79.6%.
- SpeechBrain's [wav2vec 2.0 IEMOCAP model](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP) reports 78.7% (75.3% on average), but its training recipe splits at random by default, so speakers are shared, like my pooled numbers.

Why the CNN sits near 0.60:

1. **New speakers.** With speakers shared between training and test, cnn\_final\_3 scores 0.678 instead of 0.601 (both on seeds 1 to 5).
2. **Little data for a model learned from scratch.** Each session-out fold trains on 3,205 to 3,423 utterances. cnn\_final\_3 fits them up to 0.86 training UA while validation stays near 0.64, and wider layers alone made no detectable difference in the 3-block sweep (-0.003 on validation).
3. **Overlapping classes.** Happy+excited is confused with neutral and angry, and neutral with sad. The labels themselves are uncertain: IEMOCAP's raters agree with a Fleiss' kappa of 0.48 even on turns that reached a majority, after merging happy and excited ([Busso et al. 2008](https://msp.utdallas.edu/publications/Busso_2008_5.pdf)).

**For HuBERT.** The checkpoints are [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960) (95M) and [facebook/hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k) (317M), both under the Apache 2.0 license. A first run could follow SUPERB (the model frozen, a weighted sum of its layers, a small classifier) on my exact folds and seeds, so that it compares directly with the CNNs; fine-tuning would come after.

## Code structure

Every experiment runs from one configuration file. The code expands it into one job per configuration, fold and seed, runs the jobs on the GPUs, and writes one folder per run; the analysis reads those folders.

&#91;embedded content: how a sweep runs, from the configuration file to the test report\]

- `src/affect/`: data loading and splits, features, models (CNN, CNN + Transformer, CNN + LIF), the training loop, the queue and workers, the analysis.
- `sweeps/`: one file per sweep, the settings and seeds of every experiment.
- `scripts/`: running on the GPU node; `tests/`: the checks run before every night.

**On GitHub.** The [repository](https://github.com/basinhopping/affect) holds the code, the sweep files, the tests and one notebook; the data (IEMOCAP is licensed) and the run folders stay out. The notebook, [`notebooks/cnn_3block.ipynb`](https://github.com/basinhopping/affect/blob/main/notebooks/cnn_3block.ipynb), walks the 3-block CNN from its settings to its results:

1. Settings: every setting on its own line.
2. Check: the settings are validated and hashed; the hash's first 12 characters begin the ID of every run of the configuration.
3. Sweep: the base, one change at a time, then random combinations, expanded into one job per configuration, fold and seed.
4. Results: one run's training curve, the sweep and its confirmation on validation, the test results of the two recorded configurations, and the settings the sweep changed.
