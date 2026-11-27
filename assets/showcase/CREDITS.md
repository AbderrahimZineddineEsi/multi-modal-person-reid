# Showcase credits

The warehouse footage is by **Tiger Lily** on Pexels, released under the
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
## App captures (`app_*`)

Screen recordings and screenshots of the Gradio app, made with `reid app --showcase FILE`: the app blurs every
person who is not on a hand-picked allow-list of men (default deny, see the technical report). All of them except
`app_process.*` show AI City 2024 scene 075 (footage below), processed with the `balanced` preset.

- `app_find.gif` / `.mp4`: Find tab, filters "male" then "black" upper-body colour, then Person 11's card.
- `app_search.gif` / `.mp4`: Search tab, a body photo of Person 6 cut from camera cam_0712, matched to Person 6.
- `app_clips.gif` / `.mp4`: Clips tab, a follow clip of Person 6 (4x speed, 30 s cap), planned, then played.
- `app_process.gif` / `.mp4`: Process tab, three 10 s excerpts of the Pexels warehouse clips 4293549, 4281405
  and 4284181 (above) processed into a new environment on the GTX 1050; the waiting is sped up.
- `app_overview.png`, `app_explore.png` (Person 6), `app_cameras.png` (cam_0712, Person 6 highlighted),
  `app_metrics.png`.

## AI City Challenge clips (`aicity_follow.*`, `aicity_mosaic.*`)

Footage: NVIDIA AI City Challenge 2024, Track 1, test scene 075 (synthetic people), from the
[PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) dataset, CC-BY-4.0.
Rendered from this system's own tracking output (`balanced` preset, Kaggle T4). Mosaic: scene time 4:50-5:20.
Follow clip: `reid journey --gid 6 --start 150 --end 200 --cameras cam_0710,cam_0712,cam_0702,cam_0699
--speed 3 --blur-others --thumbs 4 --width 1440` (scene time 2:30-3:20; cam_0706 left out because a tiny figure
at the far end of its view has neither a detection nor a ground-truth box). Every person except the followed man
(ID 6) is blurred with `render.blur_others` and `render.privacy_gt` (the dataset's own person boxes, so people the
detector missed are blurred too).
