# Showcase credits

All footage used in these assets is by **Tiger Lily** on Pexels, released under the
[Pexels License](https://www.pexels.com/license/) (free to use, attribution appreciated).
The clips show the same four male warehouse workers from eight angles. The clips are shown one
after another on the timeline (3 s gaps), not simultaneously, so each "camera" is a separate clip.

| Camera | Name in the demo | Source clip |
|---|---|---|
| cam_01 | Main aisle (faces) | [4293549](https://www.pexels.com/video/men-having-a-conversation-at-work-in-the-warehouse-4293549/) |
| cam_02 | Aisle A overhead | [4281405](https://www.pexels.com/video/men-working-in-a-warehouse-4281405/) |
| cam_03 | Loading bay overhead | [4281237](https://www.pexels.com/video/working-men-warehouse-job-4281237/) |
| cam_04 | Pallet zone | [4281404](https://www.pexels.com/video/men-getting-items-from-the-shelves-in-the-warehouse-4281404/) |
| cam_05 | Dock meeting point | [4284181](https://www.pexels.com/video/people-talking-in-the-warehouse-4284181/) |
| cam_06 | Supervisor desk | [4291726](https://www.pexels.com/video/men-having-a-conversation-at-work-4291726/) |
| cam_07 | Rack 12 overhead | [4281402](https://www.pexels.com/video/men-in-the-warehouse-having-a-discussion-4281402/) |
| cam_08 | Back aisle (faces) | [4293960](https://www.pexels.com/video/men-having-a-conversation-while-at-work-4293960/) |

Processing: `reid process --preset balanced --set consolidation.merge_threshold=0.35` (GTX 1050 3 GB).

## Assets

- `hero.gif`: 11.5 s excerpt (12 fps) of the follow clip of ID 1 (red cap) handing over between cameras, with the presence timeline.
- `hero.mp4`: the full 90 s follow clip of ID 1 across all eight cameras (H.264) for LinkedIn / video embeds.
- `overhead.gif`: one high-angle camera (dock meeting point) rendered with identity boxes and ID labels.
- `identities.png`: the four people found, one best crop each, with the number of cameras they were seen in.
- `app_overview.png`: Overview tab of the Gradio app (people, cameras, run statistics).
- `app_explore.png`: Explore tab for Person 1 (journey list across cameras and presence-over-time chart).
- `app_search.png`: Search tab, a body-photo query (held-out frame from one camera) matched to Person 1.
- `app_metrics.png`: Metrics tab (speed, GPU memory, time per stage).

## AI City Challenge clips (`aicity_follow.*`, `aicity_mosaic.*`)

Footage: NVIDIA AI City Challenge 2024, Track 1, test scene 075 (synthetic people), from the
[PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) dataset, CC-BY-4.0.
Rendered from this system's own tracking output (`balanced` preset, Kaggle T4), scene time 4:50-5:20 (mosaic) and
4:50-5:32 (follow clip). Every person except the followed man (ID 6) is blurred with `render.blur_others` and
`render.privacy_gt` (the dataset's own person boxes, so people the detector missed are blurred too).
