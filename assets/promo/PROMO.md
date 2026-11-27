# Sightline promo: footage and brief

Everything needed to edit a short promo video of Sightline (about 45-60 s, 16:9, 1920x1080). All clips below
are cleared for public use: every person other than the men on the allow-list is blurred. **Use only these
clips and the ones in `assets/showcase/`; do not record new footage of other people.**

The current cut is `sightline_linkedin_promo.mp4` (60 s, 1080p, shown at the end of the main README).

## Footage

| File | Length | What it shows |
|---|---|---|
| `../showcase/aicity_mosaic.mp4` | 30 s | all 16 cameras of an AI City Challenge store at once, every person tracked with an ID (others blurred, ID 6 sharp) |
| `../showcase/aicity_follow.mp4` | 14 s | one shopper (ID 6) followed from camera to camera with the same ID, side views of other cameras, timeline |
| `../showcase/hero.mp4` | 90 s | real warehouse footage (4 men, 8 cameras): one worker (ID 1) followed, face camera and overhead hand-offs |
| `take_process.mp4` | 101 s, real time | app: upload 3 videos, choose a preset, run, finished numbers and stage table (the wait is long: speed it up) |
| `take_overview.mp4` | 6 s | app: headline numbers and the people grid of the 16-camera scene |
| `take_search.mp4` | 10 s | app: a photo of a man dropped in, matched to Person 6 with where and when he was seen |
| `take_find.mp4` | 18 s | app: describe a person (Male, then black top), results narrow 46 -> 29 -> 16 people, one person opened |
| `take_clips.mp4` | 26 s | app: plan a follow clip of Person 6, render, play it (playback stops by itself; never extend past the end) |
| `take_metrics.mp4` | 4 s | app: speed and GPU memory of a run |

App takes are 1920x1080, 30 fps, recorded in real time with a visible cursor and about 1 s of hold at each end.

## Suggested beats and voiceover

| time | footage | caption | voiceover |
|---|---|---|---|
| 0:00 | mosaic | 16 cameras. Dozens of people. | Sixteen cameras. Dozens of people. Who is who? |
| 0:04.5 | follow clip | One identity across every camera | Sightline follows every person from camera to camera, with one identity, and no training on this store. |
| 0:11 | warehouse (hero) | Body appearance, confirmed by face | It matches people by how they look, and confirms them by face whenever one is visible. |
| 0:17 | take_process | Drop in footage. Press run. | Drop in any footage, press run. It even works on a laptop GPU. |
| 0:22.5 | take_overview | Everyone seen, one card each | Everyone it saw, one card per person. |
| 0:27 | take_search | Search by photo | Search with a photo... |
| 0:33 | take_find | ...or just describe them | ...or just describe them: a man in a black jacket. |
| 0:40 | take_clips | Their whole journey, in one clip | Then watch their whole journey as a single clip. |
| 0:47.5 | three number cards | (see numbers) | Fifty-five point three HOTA on unseen AI City scenes. No training. Everything stays on your machine. |
| 0:53 | logo end card | Sightline: open-world person re-identification across any cameras | Sightline. Open-world person re-identification, across any cameras. |

## Numbers that may be shown (all measured, see the main README)

- **55.3 HOTA** on 3 unseen AI City Challenge 2024 scenes (official NVIDIA evaluator), no training on the
  benchmark.
- **16 cameras** per scene; about **21 fps** on one T4; **8.5 fps** on a GTX 1050 laptop GPU.
- About **1 GB** of GPU memory for the whole pipeline (measured 0.68-1.05 GB). Do not write "under 1 GB".
- Attribute head zero-shot on Market-1501: **mA 82.6%** per person.
- 100% local, no cloud.

## Look

Dark background (#0b0f14), orange accent (#e8651a), font Instrument Sans (same as the app), logo = the app's
logo (a dashed line through coloured dots), name written "Sight" in white + "line" in orange. Captions short,
bottom centre. Neutral end card: project name and tagline only.

## Rules

- Do not use any other footage of people, and do not un-blur or enhance anyone.
- Do not show the Clips take past its end.
