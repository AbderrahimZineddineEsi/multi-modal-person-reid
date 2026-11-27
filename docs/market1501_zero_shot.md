# Zero-shot test of the attribute head on Market-1501

How well does the pedestrian-attribute head transfer to a dataset it has never seen? The head
(an MLP on top of the frozen CLIP-ReID MSMT17 body embedding, trained on PA-100K + PETA) is applied,
unchanged, to the test identities of Market-1501 and scored against the attribute annotations of
Lin et al. (2019). **Nothing is trained or tuned on Market-1501**: the decision threshold is the fixed
0.5 used everywhere else, the validation-tuned thresholds stored in the head are an extra, and the
label mapping was fixed from the annotation file and a visual check of the images before any
prediction was computed. This is a measurement of generalisation, not a benchmark claim: the
numbers are not comparable with published Market-1501 attribute accuracies (which train on
Market-1501 and use a different metric).

## Summary

- On the **21 labels** that can be mapped soundly and have at least 30 positive and 30 negative
  identities, the head reaches **mA 80.5 % per image** and **82.6 % per identity** (average of up to
  16 random crops of the identity, 20 draws) at threshold 0.5. With the thresholds tuned on the
  PA-100K / PETA validation sets: 83.5 % and 86.3 %.
- Because labels are per identity, the sampling uncertainty is that of 750 identities, not 13,115
  images: the 95 % bootstrap interval over identities is 79.4 to 81.4 % (image level) and
  81.5 to 83.8 % (identity level). The "± 0.1" in the tables below is only the noise of the random
  crop draws, not an uncertainty of the result.
- On the labels shared with each in-domain test set the zero-shot mA is **not lower** than the
  in-domain mA (PA-100K, 8 labels: 86.1 in-domain vs 85.5 on Market per image; PETA, 20 labels: 79.7
  vs 81.3). This should be read with the caveats at the end: the label definitions and the
  protocol differ, and part of the gap in either direction comes from that.
- Clothing and carried-object labels transfer well (mean mA 85.0 over 9 labels, AUC above 90 for 16
  of the 21 labels). **Colours are the weak group**, lower-body colours in particular (mean mA 72.6
  over 5 labels, against 80.3 for 7 upper-body colours). The worst labels are `lower_blue` (58.3),
  `handbag` (65.2), `lower_brown` (69.8) and `upper_grey` (70.4).

## What was evaluated

| | |
|---|---|
| Head | `head.npz` trained on PA-100K + PETA (hidden 512, tuned hyper-parameters, hash `88d596018fa860b5`), the one `reid attributes use-head` installs |
| Backbone | CLIP-ReID ViT-B/16 MSMT17, frozen, `after` neck, 1280-D; same code path and preprocessing as the pipeline and as the training embeddings; no flipped copies |
| Dataset | Market-1501 `bounding_box_test`, 750 test identities |
| Labels | Lin et al. 2019 attributes (`market_attribute.mat`), annotated per identity |
| Metrics | mA, instance Acc / Prec / Rec / F1 (the definitions of `reid attributes report`), per-label recalls and ROC AUC |
| Reference | the head's own test report (PA-100K official test split, PETA person-disjoint 40 % test split), read from the existing report file, not recomputed |

### Dataset statistics

- `bounding_box_test` holds 19,732 images. 3,819 are junk (identity `-1`) and 2,798 are distractors
  (identity `0000`); neither has attributes and both are dropped. **13,115 images of 750 identities
  remain.** The 3,368 `query` images (further crops of the same identities) are not used, so
  every image is scored once.
- Crops per identity: minimum 2, median 16, mean 17.5, maximum 71; 360 identities have fewer than
  16 crops (they use all of them in the identity-level vote).
- Every crop is 128 x 64 pixels (checked on 300 random crops: minimum equals maximum). The
  same check on 300 random PA-100K images gave a median of 197 x 73 pixels (range 118 x 50 to
  598 x 242). The backbone input is 256 x 128, so Market crops are enlarged about 2x.
- The annotation file has 750 test identities and they are exactly the identities of the 13,115
  images. All attributes use 1 / 2 (not 0 / 1).

## Label mapping

Rules live in `src/reid/attributes/schema.py` (`MARKET_*`, `market_targets`). Market values are
1 = no / 2 = yes unless stated. Encodings were verified on images of the test identities (for example
gender 1 is male, `up` 1 is long sleeve, `clothes` 1 is a dress).

| Market column(s) | our label | rule | note |
|---|---|---|---|
| `gender` | `female` | 2 (female) -> 1, 1 -> 0 | |
| `hair` | `long_hair` | 2 (long) -> 1, 1 (short) -> 0 | |
| `up` | `short_sleeve`, `long_sleeve` | 2 -> short = 1, long = 0; 1 the opposite | the two are exclusive in Market |
| `clothes`, `down` | `trousers` | pants and long | includes jeans, like our PETA rule |
| `clothes`, `down` | `shorts` | pants and short | "short" has no stated length limit (capri / knee-length are unspecified) |
| `clothes` | `skirt_dress` | dress -> 1, pants -> 0 | |
| `hat` | `hat` | 2 -> 1 | only 23 identities: excluded from mA |
| `backpack` | `backpack` | 2 -> 1 | identity-level: see analysis |
| `handbag` | `handbag` | 2 -> 1 | Market's `bag` ("other types of bag") is a separate class and stays unmapped, so the handbag / bag boundary of the annotators is unknown |
| `upblack`, `upwhite`, `upred`, `uppurple`, `upyellow`, `upgray`, `upblue`, `upgreen` | `upper_black` ... `upper_green` (8) | 2 -> 1, 1 -> 0. Exactly one colour per identity is 2, or none (69 identities, a colour outside the list): then all eight are 0 | one colour per identity, so a two-coloured top has only one of its colours as positive |
| `downblack`, `downwhite`, `downpurple`, `downyellow`, `downgray`, `downblue`, `downgreen`, `downbrown` | `lower_black` ... `lower_brown` (8) | same rule | `downbrown` positives look brown / khaki / beige |

**Not mapped (unsure) and why**

- `age` (young / teenager / adult / old): the paper and the README name the classes differently
  (child / teenager / adult / old vs young / teenager / adult / old) and give no age ranges.
  640 of the 750 identities are "teenager", so no boundary to our child (< 18) / adult / senior
  (> 60) can be justified. The class counts (2 young, 5 old) are also far too small to score.
- `bag` -> `shoulder_bag`: the paper defines `bag` as "other types of bag", which is not
  "shoulder bag", and the shoulder-bag / handbag boundary differs between the datasets.
- `downpink` -> `lower_pink`: the positives mix pink and red clothes (the paper lists "red" where
  the file says "pink"), and there is no matching schema label. For the lower colours that do exist,
  an identity labelled `downpink` counts as 0.
- Our labels with no Market source (view, glasses, scarf, hair colour, coat / jacket / t-shirt /
  sweater, patterns, shoes, plastic bag, suitcase, stroller, `upper_brown`, `upper_orange`,
  `upper_pink`, `lower_orange`, `lower_pink`, `lower_red`, ...) stay unknown (-1) and are not scored.

The paper and the annotation file also disagree on the colour lists (the paper names brown for the
upper body and red for the lower body; the file has yellow and pink). The file is what is scored.

**Labels in the mapping but out of the headline mA** (fewer than 30 positive or negative
identities, counted per identity because identities are the independent units): `hat` (23
positives), `upper_purple` (24), `lower_green` (20), `lower_yellow` (1), `lower_purple` (0). The
mapped but excluded labels are still scored in the "all mapped" column (26 labels): 76.9 % per image,
78.4 % per identity. That is 21 labels in the headline and 5 outside it.

## Results

All numbers are percentages. Threshold 0.5. Reproduce with the command at the end.

### Headline

| level | mA | Acc | Prec | Rec | F1 | mA (val-tuned thr.) | mA (all mapped) |
|---|---|---|---|---|---|---|---|
| image (13115) | 80.5 | 67.5 | 80.2 | 78.7 | 79.4 | 83.5 | 76.9 |
| identity, 16 crops (750, 20 draws) | 82.6 ± 0.1 | 73.2 ± 0.1 | 85.9 ± 0.1 | 81.5 ± 0.1 | 83.6 ± 0.1 | 86.3 ± 0.1 | 78.4 ± 0.1 |

95 % interval of mA over resampled identities (the independent units; 500 resamples): image level 79.4 to 81.4, identity level 81.5 to 83.8.

"mA (val-tuned thr.)" uses the per-label thresholds stored in the head (tuned on the PA-100K / PETA
validation sets, never on Market). "mA (all mapped)" adds the five rare labels. Identity level
averages the probabilities of up to 16 random crops of each identity (all of them if the identity
has fewer), mean over 20 draws with a fixed seed.

At threshold 0.5 the head is conservative on Market: over the 21 labels the mean recall of
positives is 68.4 % and of negatives 92.6 % (per image). The lower thresholds from validation
move this balance and add 3 points of mA.

### Same labels, in-domain vs Market

| in-domain test set | labels in common | in-domain mA | Market, image level | Market, identity level |
|---|---|---|---|---|
| pa100k | 8 | 86.1 | 85.5 | 88.9 |
| peta | 20 | 79.7 | 81.3 | 83.4 |

The in-domain numbers are per image, from the head's own test report. The identity-level column
uses a different protocol (voting) and is shown only to quantify what the vote gives.

### Per label

| label | pos / neg ids | pos / neg images | image mA | pos rec | neg rec | AUC | identity mA | pa100k mA | peta mA | drop |
|---|---|---|---|---|---|---|---|---|---|---|
| female | 336 / 414 | 5781 / 7334 | 92.8 | 92.2 | 93.4 | 97.3 | 96.2 | 91.8 | 88.0 | -2.9 |
| long_hair | 275 / 475 | 4732 / 8383 | 81.6 | 65.8 | 97.4 | 95.6 | 81.6 | - | 78.6 | -3.0 |
| short_sleeve | 705 / 45 | 12267 / 848 | 84.7 | 93.0 | 76.4 | 93.5 | 87.6 | 89.9 | 87.4 | +3.9 |
| long_sleeve | 45 / 705 | 848 / 12267 | 84.5 | 75.5 | 93.6 | 93.8 | 87.3 | 89.9 | 88.9 | +4.9 |
| upper_black | 109 / 641 | 1759 / 11356 | 87.0 | 78.0 | 95.9 | 96.4 | 90.8 | - | 88.5 | +1.5 |
| upper_blue | 52 / 698 | 1100 / 12015 | 74.8 | 51.6 | 97.9 | 91.7 | 75.5 | - | 71.1 | -3.7 |
| upper_green | 45 / 705 | 949 / 12166 | 85.2 | 72.6 | 97.7 | 93.7 | 85.6 | - | 72.2 | -12.9 |
| upper_grey | 106 / 644 | 1755 / 11360 | 70.4 | 46.0 | 94.9 | 81.1 | 73.1 | - | 79.7 | +9.3 |
| upper_red | 76 / 674 | 1354 / 11761 | 79.5 | 60.0 | 99.1 | 95.0 | 83.5 | - | 91.3 | +11.7 |
| upper_white | 208 / 542 | 3491 / 9624 | 86.3 | 91.4 | 81.3 | 92.3 | 88.6 | - | 81.6 | -4.8 |
| upper_yellow | 61 / 689 | 1195 / 11920 | 78.8 | 58.0 | 99.6 | 94.6 | 77.2 | - | 72.8 | -6.0 |
| trousers | 252 / 498 | 4142 / 8973 | 92.9 | 93.4 | 92.3 | 97.3 | 95.4 | 91.3 | 74.7 | -9.9 |
| shorts | 413 / 337 | 7408 / 5707 | 88.7 | 87.3 | 90.2 | 95.3 | 93.3 | 91.1 | 75.2 | -5.6 |
| skirt_dress | 85 / 665 | 1565 / 11550 | 88.9 | 82.6 | 95.2 | 97.2 | 93.9 | 83.8 | 69.1 | -12.4 |
| lower_black | 287 / 463 | 5065 / 8050 | 82.5 | 75.8 | 89.3 | 91.2 | 87.0 | - | 85.2 | +2.6 |
| lower_blue | 154 / 596 | 2632 / 10483 | 58.3 | 18.4 | 98.2 | 79.1 | 54.9 | - | 79.9 | +21.6 |
| lower_brown | 48 / 702 | 907 / 12208 | 69.8 | 41.9 | 97.8 | 89.8 | 70.0 | - | 78.8 | +8.9 |
| lower_grey | 126 / 624 | 2287 / 10828 | 74.7 | 78.0 | 71.3 | 82.6 | 78.9 | - | 74.2 | -0.5 |
| lower_white | 50 / 700 | 711 / 12404 | 77.9 | 61.6 | 94.3 | 93.2 | 76.2 | - | 75.7 | -2.2 |
| backpack | 187 / 563 | 3291 / 9824 | 85.9 | 77.2 | 94.6 | 92.7 | 91.4 | 79.4 | 81.4 | -5.5 |
| handbag | 76 / 674 | 1242 / 11873 | 65.2 | 37.3 | 93.0 | 77.6 | 66.2 | 71.5 | - | +6.4 |

"pos / neg ids" counts identities, "pos / neg images" counts crops. `pa100k mA` / `peta mA` are the
in-domain test mA of the label (only where the label exists there and has at least 30 test
positives). **drop** = mean of the available in-domain mA minus the Market image-level mA, in
points: positive means the head is worse on Market. Labels with 45 to 76 positive identities have
a standard error of about 3 mA points (binomial approximation), so differences of a few points
between labels are not meaningful.

### Voting over crops of one identity

| k crops | identities with >= k crops | mA (k crops) | mA (1 crop, same identities) |
|---|---|---|---|
| 1 | 750 | 80.3 ± 0.4 | 80.4 |
| 2 | 750 | 81.5 ± 0.3 | 80.3 |
| 4 | 739 | 82.3 ± 0.3 | 80.6 |
| 8 | 652 | 82.4 ± 0.2 | 80.7 |
| 16 | 390 | 82.0 ± 0.3 | 80.2 |
| 38 | 31 | 88.2 ± 1.0 | 82.1 |

The set of identities changes with k (only identities with at least k crops), which is why the
table gives the one-crop mA on the same identities. Voting gains about 2 points and saturates between 4
and 8 crops. The last row has only 31 identities and is noisy. Tracklets in the pipeline are
more correlated than random crops of an identity, so the real gain inside the pipeline is likely
smaller (this was not measured here).

### Agreement between the crops of one identity

| label | positive ids: crops predicted positive | positive ids with mixed crops | negative ids: crops predicted positive | negative ids with mixed crops |
|---|---|---|---|---|
| female | 91.2 | 32.4 | 6.7 | 42.3 |
| long_hair | 63.4 | 70.2 | 2.7 | 15.4 |
| short_sleeve | 92.3 | 38.3 | 25.1 | 73.3 |
| long_sleeve | 73.7 | 71.1 | 7.0 | 35.9 |
| upper_black | 78.1 | 61.5 | 4.3 | 17.3 |
| upper_blue | 49.8 | 82.7 | 1.9 | 8.3 |
| upper_green | 74.5 | 64.4 | 2.5 | 8.7 |
| upper_grey | 47.1 | 73.6 | 5.3 | 23.0 |
| upper_red | 63.8 | 36.8 | 0.9 | 5.5 |
| upper_white | 91.3 | 41.8 | 19.3 | 39.1 |
| upper_yellow | 56.4 | 57.4 | 0.3 | 0.9 |
| trousers | 93.6 | 28.2 | 8.8 | 40.2 |
| shorts | 85.7 | 56.9 | 9.1 | 33.5 |
| skirt_dress | 83.3 | 68.2 | 5.2 | 23.2 |
| lower_black | 75.6 | 60.6 | 11.0 | 35.6 |
| lower_blue | 17.8 | 56.5 | 1.6 | 9.1 |
| lower_brown | 42.3 | 81.2 | 1.8 | 11.8 |
| lower_grey | 78.7 | 68.3 | 28.2 | 60.6 |
| lower_white | 58.7 | 90.0 | 5.9 | 21.6 |
| backpack | 78.9 | 62.0 | 6.2 | 31.4 |
| handbag | 39.4 | 90.8 | 7.2 | 35.0 |

For each label: the share of crops predicted positive among the crops of positive identities (it
equals the per-crop recall), the share of positive identities whose crops disagree (some above
and some below 0.5), and the same two for negative identities. Disagreement between crops is
expected when a label is only visible from some viewpoints; it can also come from model error, so
it is evidence of view-dependence, not proof of label noise.

## Analysis

**Largest drops** (mean in-domain mA minus Market image mA, points):
`lower_blue` +21.6, `upper_red` +11.7, `upper_grey` +9.3, `lower_brown` +8.9, `handbag` +6.4,
`long_sleeve` +4.9, `short_sleeve` +3.9. The causes below are what the data shows; none was isolated
by an ablation, so they are likely explanations, not proven ones.

- **`lower_blue` (58.3, positive recall 18.4 %).** For 99 of the 154 positive identities the head's
  highest lower-body colour is *grey* (30 black, 22 blue). Among the 12 lowest-scored positive
  identities (viewed by eye) several wear denim shorts or skirts, others a blue or teal dress
  (where the "lower" colour is the dress colour); several look grey or dark at this resolution. The head separates the label
  reasonably (AUC 79.1) but at threshold 0.5 almost no positive crosses it; the validation-tuned
  threshold of 0.19 recovers part of it (mA 64.2 instead of 58.3).
- **`upper_grey` (70.4) and `upper_red` (79.5).** For 41 of the 106 grey-top identities the head's
  highest upper colour is white (light grey against white). Among the 12 lowest-scored `upper_red`
  identities most tops are pink, salmon, maroon or purple. Market has no pink or orange upper class, so
  such tops are labelled red. The head's own `upper_pink` score for the red positives is low as well
  (mean identity probability 0.15), so this is not only a definition difference. `lower_brown` (69.8) shows the
  same pattern: for 26 of the 48 positive identities the highest lower colour is grey (khaki and
  beige trousers against grey).
- **`handbag` (65.2, AUC 77.6).** The 12 lowest-scored positives carry small hand-held
  items (small bags, folders, paper or plastic bags) at 128 x 64, and in several of them no bag is
  clearly visible in the crop shown. Only 39 % of the crops of positive identities
  are predicted positive and 91 % of positive identities have mixed crops. On PA-100K the
  label is also the lowest of the eight labels shared with Market (71.5), so it is hard in
  general.
- **`short_sleeve` / `long_sleeve` (84.7 / 84.5, against 89.9 on PA-100K).** 705 of the 750
  identities (94 %) have short sleeves, so the negative class of `short_sleeve` has only 45
  identities. A quarter of the crops of long-sleeve identities are
  predicted short-sleeve (25.1 % false positive rate), while 7 % of the crops of short-sleeve
  identities are predicted long-sleeve.
- **`backpack`.** 85.9 per image against 79.4 on PA-100K and 81.4 on PETA, so it does *not*
  drop. It does show the effect of identity-level labels: the authors annotate a backpack as
  present for an identity even when one image does not show it (stated in the paper). Only 78.9 % of
  the crops of backpack identities are predicted positive and 62 % of them have mixed crops; the
  identity-level vote lifts mA from 85.9 to 91.4.

**Labels where Market is easier than the in-domain set.** `upper_green` (+12.9 against PETA 72.2),
`skirt_dress` (88.9 against 83.8 on PA-100K and 69.1 on PETA), `trousers` (92.9 against 91.3 and
74.7) and `female` (92.8 against 91.8 and 88.0) are at or above their in-domain level. PETA is the
harder in-domain set (the head's overall test mA is 71.6 on PETA against 77.9 on PA-100K), so a
label scored higher on Market than on PETA says little about transfer.

**Why not a bigger drop?** Consistent with the evidence, not proven: Market labels are coarse
(gender, sleeve length, one colour) and identity-level, the mA metric is balanced over the two
classes, and the voting over many crops removes a large part of the per-crop noise.

## Caveats

- **Effective sample size is about 750.** Labels are per identity, copied to every crop. The
  13,115 crops of one identity are strongly correlated, so they do not count as 13,115 independent
  test samples. Use the identity bootstrap interval, not the crop-draw ± values.
- **Not a fully foreign domain.** The body backbone is the MSMT17 checkpoint: a large pedestrian
  re-identification set recorded with surveillance-style cameras on a campus, so viewpoints and crop
  style resemble Market-1501's. Market-1501 itself is not part of the training of the backbone
  (as named) or of the head, but the domain gap is likely smaller than for an unrelated source.
  How much this helps is untested here.
- **Market was never used for any decision.** No threshold, label weight, hyper-parameter or
  model choice looked at Market predictions. The mapping was fixed from the annotation file and
  a visual check of the images (not of the head's output) before the head was run; the diagnosis
  in the analysis section was written after the run and changed nothing.
- **Definitions differ.** One colour per identity (Market) against multi-hot colours (PETA);
  the `bag` / `handbag` boundary; `shorts` without a length limit; `lower_brown` including khaki.
  Part of every per-label gap comes from this and cannot be separated from model error here.
- **Different test populations and protocols.** The in-domain numbers come from other label sets
  and splits (PETA with our person-disjoint split). Per-label comparisons are indicative; a
  difference of a few points is within the noise of 45 to 100 identities.
- **Resolution.** All Market crops are 128 x 64 against a median of 197 x 73 on PA-100K, which is
  the usual suspect for colour confusions and small objects. No resolution ablation was run, so
  this stays a hypothesis.
- **Numerics.** The training embeddings were computed on a GPU with fp16; the Market embeddings
  here with fp32 (the local GPU has no fast half). On 256 PA-100K test images that have both, the
  cosine similarity is above 0.9999 and 0.01 % of the head's decisions at 0.5 differ.
- **Data and licence.** Market-1501 is for research only ("Please DO NOT distribute or use it
  for commercial purpose") and is never copied into this repository. The attribute file is from
  the authors' public repository (`vana77/Market-1501_Attribute`).

## Reproduce

Download Market-1501 from its [official page](https://zheng-lab-anu.github.io/Project/project_reid.html)
(Google Drive / Baidu links) and `market_attribute.mat` from the
[attribute repository](https://github.com/vana77/Market-1501_Attribute). Unzip the dataset into its own
empty folder (anything below `data/` is git-ignored), then:

```bash
reid attributes eval-external --dataset market1501 \
    --root data/datasets/market1501 \
    --attribute-file data/datasets/market1501_attr_dl/market_attribute.mat \
    --out data/attributes/market1501_zero_shot \
    --in-domain data/attributes/<head run>/report/metrics_<tag>.json
```

`--head` defaults to the installed head (`reid attributes use-head`). The body model needs a GPU
for reasonable speed (13,115 images took 9.3 minutes at 23 images/s on a GTX 1050; the same command
works on Kaggle). Embeddings are resumable and stored in `<out>/emb/market1501/test`; a second run
reuses them and finishes in a few seconds. The command writes `RESULTS_market1501.md` (the full
tables) and `metrics_market1501.json`.

References: Lin et al., *Improving Person Re-identification by Attribute and Identity Learning*,
Pattern Recognition 2019; Zheng et al., *Scalable Person Re-identification: A Benchmark*, ICCV 2015.
