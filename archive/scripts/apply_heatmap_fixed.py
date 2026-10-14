import json

with open("module6_dashboard.ipynb", "r") as f:
    nb = json.load(f)

# Find cell 11 (index 10) which ends with print("✅ Pipeline & search functions defined.")
cell_11 = nb["cells"][10]
source_11 = "".join(cell_11["source"])

heatmap_funcs = """

# ═══════════════════════════════════════════════════════════════
# HEATMAP (person positions on a fixed camera-frame background)
# ═══════════════════════════════════════════════════════════════
def _bbox_centers_from_sightings(sightings):
    \"\"\"Return list of (cx, cy, x1, y1, x2, y2) in pixel coords from `best_crop_bbox`.\"\"\"
    pts = []
    for s in sightings:
        bb = s.get("best_crop_bbox")
        if not bb or len(bb) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bb]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        pts.append((cx, cy, x1, y1, x2, y2))
    return pts

def load_default_heatmap_background():
    \"\"\"Load the fixed camera-frame image used as the heatmap background from ImageKit.\"\"\"
    import cv2, urllib.request, numpy as np
    
    # We strip trailing slash from IMAGEKIT_URL if necessary, but assume it's formatted
    base_url = IMAGEKIT_URL.rstrip('/')
    bg_url = f"{base_url}/backgrounds/{CAMERA_ID}.jpg"
    
    img_rgb = fetch_image_from_url(bg_url, as_rgb=True)
    if img_rgb is None:
        return None, f"Could not load heatmap background: `{bg_url}` \\nMake sure to upload a fixed background picture named `{CAMERA_ID}.jpg` inside a `backgrounds` folder in ImageKit."
    
    return img_rgb, ""

def build_person_heatmap(sightings, background_rgb=None,
                         sigma_small=4, heat_scale=0.10, alpha=0.6,
                         draw_centers=True):
    \"\"\"
    Render a heatmap of a person's bbox-center positions over the fixed camera frame.
    \"\"\"
    import numpy as np
    import cv2
    pts = _bbox_centers_from_sightings(sightings)
    if not pts:
        return None, "No bbox positions available for this person."

    if background_rgb is not None and background_rgb.size > 0:
        bg = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2BGR).copy()
        H, W = bg.shape[:2]
        # remove trailing slash just in case
        base_url = IMAGEKIT_URL.rstrip('/')
        bg_url = f"{base_url}/backgrounds/{CAMERA_ID}.jpg"
        scale_note = f"(using fixed camera frame `{bg_url}` at {W}x{H})"
    else:
        max_x = max(p[4] for p in pts)
        max_y = max(p[5] for p in pts)
        W = int(max(640, max_x * 1.05))
        H = int(max(480, max_y * 1.05))
        bg = np.full((H, W, 3), 32, dtype=np.uint8)
        scale_note = f"(fallback canvas {W}x{H}; fixed camera frame could not be loaded)"

    Hs = max(32, int(H * heat_scale))
    Ws = max(32, int(W * heat_scale))
    heat_small = np.zeros((Hs, Ws), dtype=np.float32)

    stamp_r = max(2, int(round(sigma_small * 1.1)))

    for (cx, cy, *_rest) in pts:
        x = int(np.clip(round(cx * (Ws - 1) / max(1, W - 1)), 0, Ws - 1))
        y = int(np.clip(round(cy * (Hs - 1) / max(1, H - 1)), 0, Hs - 1))

        y0 = max(0, y - stamp_r)
        y1s = min(Hs - 1, y + stamp_r)
        x0 = max(0, x - stamp_r)
        x1s = min(Ws - 1, x + stamp_r)

        for yy in range(y0, y1s + 1):
            dy = yy - y
            for xx in range(x0, x1s + 1):
                dx = xx - x
                if dx * dx + dy * dy <= stamp_r * stamp_r:
                    heat_small[yy, xx] += 1.0

    k = int(max(3, (sigma_small * 6) | 1))
    heat_small = cv2.GaussianBlur(heat_small, (k, k), sigmaX=sigma_small, sigmaY=sigma_small)
    heat = cv2.resize(heat_small, (W, H), interpolation=cv2.INTER_LINEAR)
    heat_norm = heat / (heat.max() + 1e-9)
    heat_u8 = (heat_norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(bg, 1.0 - alpha, heat_color, alpha, 0)

    if draw_centers:
        cx, cy, bx1, by1, bx2, by2 = pts[-1]
        cv2.circle(overlay, (int(cx), int(cy)), max(6, int(0.005 * max(H, W))),
                   (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.rectangle(overlay,
                      (max(0, int(bx1)), max(0, int(by1))),
                      (min(W - 1, int(bx2)), min(H - 1, int(by2))),
                      (0, 255, 255), thickness=max(1, int(0.002 * max(H, W))),
                      lineType=cv2.LINE_AA)

    info = f"Heatmap built from **{len(pts)}** sighting position(s) {scale_note}."
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), info
"""

if "def load_default_heatmap_background" not in source_11:
    source_11 = source_11.replace('print("✅ Pipeline & search functions defined.")', heatmap_funcs + '\nprint("✅ Pipeline & search functions defined.")')
    cell_11["source"] = [s + '\n' for s in source_11.split('\n')]
    if cell_11["source"][-1] == '\n':
        cell_11["source"].pop()

cell_13 = nb["cells"][12]
source_13 = "".join(cell_13["source"])

# 1. Update run_search def
source_13 = source_13.replace(
    "def run_search(image, image_url, search_mode, is_face_crop, body_threshold, face_threshold, tau):",
    "def run_search(image, image_url, search_mode, is_face_crop, body_threshold, face_threshold, tau,\n               heatmap_alpha=0.6, heatmap_sigma=5):"
)
source_13 = source_13.replace(
    "return \"Please upload an image or provide a URL.\", None, \"\"",
    "return \"Please upload an image or provide a URL.\", None, \"\", None, \"\""
)
source_13 = source_13.replace(
    "return f\"Failed to load image from URL: {e}\", None, \"\"",
    "return f\"Failed to load image from URL: {e}\", None, \"\", None, \"\""
)

# 2. Update run_search return block
old_return = """    if body_result and body_result["status"] == "MATCH": best_gid = body_result["global_id"]
    if face_result and face_result["status"] == "MATCH": best_gid = face_result["global_id"]
    if best_gid:
        info, _, _ = get_person_details(best_gid)
        details = info

    return "\\n\\n".join(lines), display_img, details"""

new_return = """    if body_result and body_result["status"] == "MATCH": best_gid = body_result["global_id"]
    if face_result and face_result["status"] == "MATCH": best_gid = face_result["global_id"]

    heatmap_img = None
    heatmap_info = ""
    if best_gid is not None:
        info, _, _ = get_person_details(best_gid)
        details = info

        # Build heatmap on the fixed camera-frame background.
        person_doc = persons_col.find_one({"global_id": int(best_gid)}, {"_id": 0, "sightings": 1})
        sightings = (person_doc or {}).get("sightings", [])
        camera_bg_rgb, bg_warning = load_default_heatmap_background()
        heatmap_img, heatmap_info = build_person_heatmap(
            sightings,
            background_rgb=camera_bg_rgb,
            sigma_small=int(heatmap_sigma),
            alpha=float(heatmap_alpha),
        )
        if bg_warning:
            heatmap_info = f"{bg_warning}\\n\\n{heatmap_info}"

    return "\\n\\n".join(lines), display_img, details, heatmap_img, heatmap_info"""

if "heatmap_img, heatmap_info" not in source_13:
    source_13 = source_13.replace(old_return, new_return)


# 3. Update the UI layout
old_ui1 = """                    with gr.Accordion("Search Settings", open=False):
                        body_th = gr.Slider(0.3, 1.0, value=0.75, step=0.05, label="Body Threshold")
                        face_th = gr.Slider(0.3, 1.0, value=0.55, step=0.05, label="Face Threshold")
                        search_tau = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label="Tau")

                    search_btn = gr.Button("Search", variant="primary")"""

new_ui1 = """                    with gr.Accordion("Search Settings", open=False):
                        body_th = gr.Slider(0.3, 1.0, value=0.75, step=0.05, label="Body Threshold")
                        face_th = gr.Slider(0.3, 1.0, value=0.55, step=0.05, label="Face Threshold")
                        search_tau = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label="Tau")
                    with gr.Accordion("Heatmap Settings", open=False):
                        heat_alpha = gr.Slider(0.1, 1.0, value=0.6, step=0.05, label="Map Opacity")
                        heat_sigma = gr.Slider(2, 10, value=5, step=1, label="Detection Size")

                    search_btn = gr.Button("Search", variant="primary")"""

source_13 = source_13.replace(old_ui1, new_ui1)

old_ui2 = """            with gr.Column(scale=1):
                gr.Markdown("### Search Results")
                search_out = gr.Markdown()
                search_img = gr.Image(label="Matched Image")
                search_hist = gr.Markdown(label="Identity History")"""

new_ui2 = """            with gr.Column(scale=1):
                gr.Markdown("### Search Results")
                search_out = gr.Markdown()
                search_img = gr.Image(label="Matched Image")
                search_hist = gr.Markdown(label="Identity History")
                heat_img = gr.Image(label="Person Heatmap")
                heat_info = gr.Markdown()"""

source_13 = source_13.replace(old_ui2, new_ui2)

old_btn = """            search_btn.click(
                fn=run_search,
                inputs=[query_img, query_url, search_mode, is_face_crop, body_th, face_th, search_tau],
                outputs=[search_out, search_img, search_hist]
            )"""

new_btn = """            search_btn.click(
                fn=run_search,
                inputs=[query_img, query_url, search_mode, is_face_crop, body_th, face_th, search_tau, heat_alpha, heat_sigma],
                outputs=[search_out, search_img, search_hist, heat_img, heat_info]
            )"""

source_13 = source_13.replace(old_btn, new_btn)

def to_notebook_lines(text):
    lines = text.split('\n')
    return [l + '\n' for l in lines[:-1]] + [lines[-1]] if lines else []

cell_13["source"] = to_notebook_lines(source_13)

# Update build_results_html to add heatmap_img as argument and insert it (wait, the prompt just said "apply it in module6_dashboard.ipynb"). The diff earlier showed my friend also added heatmap handling to build_results_html but that actually wasn't in build_results_html, it was in `run_video_handler`? Let's check `heatmap.py`. From my earlier diff, my friend replaced it in `run_search`. Wait! The previous diff patched `run_video_handler` only to pass down `report = build_results_html()`. 
# Looking at the diff, my friend patched `run_search()`! My modifications above handle that!

with open("module6_dashboard.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Done patching module6_dashboard.ipynb")
