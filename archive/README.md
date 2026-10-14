# Archive

Superseded versions and development scripts, kept for reference only.
Nothing here is part of the final pipeline — the current versions live in `notebooks/`.

## notebooks/

| File | Why archived | Superseded by |
|------|--------------|---------------|
| `fork-of-reid-module4-face-association.ipynb` | Early fork of Module 4 | `notebooks/module4_face_association.ipynb` |
| `reid-module3-prototype.ipynb` | Standalone α-FPS prototype search engine (Module 3) | `notebooks/module5_search_engine.ipynb` (adds face search) |
| `module6_dashboard_old.ipynb` | First dashboard, no heatmaps, simpler tracker | `notebooks/module6_dashboard.ipynb` |
| `module6_dashboard_vfinal_with_heatmap.ipynb` | Earlier single-camera heatmap version, older BotSort workaround | `notebooks/module6_dashboard.ipynb` (multi-camera heatmaps + robust tracker) |

## scripts/

Development / one-off helper scripts that were used to iterate on the notebooks.
They are not needed to run the pipeline.

| File | Purpose |
|------|---------|
| `current.py` | Code-cell export of an intermediate dashboard revision |
| `heatmap.py` | Code-cell export including the first heatmap implementation |
| `apply_heatmap.py` | Script that injected heatmap functions into the dashboard notebook |
| `apply_heatmap_fixed.py` | Follow-up imagekit-backed heatmap injection |
| `temp_run_search.py` | Scratch copy of the search handler while debugging |
| `diff.patch` | Diff between `current.py` and `heatmap.py` |

## docs/

| File | Why archived |
|------|--------------|
| `pipeline_technical_report.md` | Earlier, shorter technical report | superseded by `../docs/technical_report.md` |
