"""Search tab: find a person from a photo by body, face or both."""

from __future__ import annotations

import gradio as gr

from reid import search
from reid.app import clips_data, views
from reid.app import components as C
from reid.app.media import array_uri, fetch_image, rgb_to_bgr
from reid.app.state import AppContext
from reid.storage import db
from reid.types import SearchHit

MODES = {"Body": search.search_body, "Face": search.search_face, "Both": search.search_both}
MODE_HELP = {
    "Body": "Compares clothing and build. Works with any photo of a full person.",
    "Face": "Compares the face only. Needs a clear, frontal face.",
    "Both": "Tries the face first and falls back to the body.",
}


def result_html(ctx: AppContext, conn, hit: SearchHit, query_uri: str) -> str:
    """Result card: status, query next to the best match, ranked candidates, journey."""
    status = hit.status.value
    head = C.status_badge(status) + C.badge(f"by {hit.modality}", "dim")
    if hit.gid is not None:
        head += C.badge(f"similarity {hit.score:.2f}", "dim")
    parts = [f'<div class="rd-result-top">{head}</div>']
    if status == "NO_FACE":
        parts.append(C.empty_state("No face found", "Use Body or Both, or try a closer photo."))
    elif hit.gid is None:
        parts.append(C.empty_state("Nobody in this database looks like the photo"))
    else:
        card = next(c for c in views.people_cards(ctx, conn) if c.gid == hit.gid)
        if status == "AMBIGUOUS" and hit.second_gid is not None:
            parts.append(
                f'<p class="rd-hint">Person {hit.gid} is the best guess, but '
                f"Person {hit.second_gid} scored almost as high.</p>"
            )
        parts.append(C.identity_card(card, big=True))
        parts.append(C.journey_html(hit.gid, views.journey_steps(ctx, conn, hit.gid)))
    names = views.person_names(conn)
    parts.append(C.candidates_html(hit.candidates[:5], names, hit.gid))
    query = f'<img class="rd-query" src="{query_uri}" alt="Your photo">' if query_uri else ""
    return f'<div class="rd-result">{query}{"".join(parts)}</div>'


def run_search(ctx: AppContext, image, url: str, mode: str, is_crop: bool, progress):
    """Returns (result html, found gid, visibility of the explore button)."""
    if image is not None:
        img = rgb_to_bgr(image)
    elif url and url.strip():
        try:
            img = fetch_image(url)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc
    else:
        raise gr.Error("Upload a photo or paste an image URL.")
    with ctx.db() as conn:
        if not db.list_identities(conn):
            raise gr.Error("Nobody to search yet. Process some footage first.")
        progress(0.2, desc="Reading the photo (first search loads the models)")
        kw = {"is_crop": True} if is_crop and mode != "Body" else {}
        hit = MODES[mode](img, conn, ctx.vectors(), ctx.cfg, models=ctx.models(), **kw)
        html = result_html(ctx, conn, hit, array_uri(img))
    return html, hit.gid, gr.update(visible=hit.gid is not None)


def build(ctx: AppContext) -> gr.Tab:
    with gr.Tab("Search", id="search"):
        found = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                image = gr.Image(
                    label="Photo", sources=["upload", "clipboard"], type="numpy", height=300
                )
                url = gr.Textbox(
                    label="Or an image URL", placeholder="https://example.com/person.jpg"
                )
                mode = gr.Radio(list(MODES), value="Both", label="Search by")
                hint = gr.Markdown(MODE_HELP["Both"], elem_classes="rd-hint")
                crop = gr.Checkbox(
                    label="The photo is already a cropped face",
                    info="Skips face detection. Used by Face and Both.",
                )  # fmt: skip
                go = gr.Button("Search", variant="primary", size="lg")
            with gr.Column(scale=3):
                result = gr.HTML(
                    C.empty_state("Results appear here", "Add a photo and press Search.")
                )
                with gr.Row():
                    to_explore = gr.Button("Open in Explore", visible=False)
                    to_clips = gr.Button("Make a clip", visible=False)

    def go_click(img, link, m, c, progress=gr.Progress()):  # noqa: B008 (Gradio's progress idiom)
        return run_search(ctx, img, link, m, c, progress)

    mode.change(lambda m: MODE_HELP[m], mode, hint)
    go.click(
        go_click, [image, url, mode, crop], [result, found, to_explore],
        concurrency_limit=1, concurrency_id="gpu",
    ).then(lambda g: gr.update(visible=g is not None), found, to_clips)  # fmt: skip
    to_clips.click(
        lambda g: (clips_data.make_prefill(g), gr.Tabs(selected="clips")),
        found,
        [ctx.ui["clips_prefill"], ctx.ui["tabs"]],
    )
    to_explore.click(
        lambda g: (g, gr.Tabs(selected="explore")),
        found,
        [ctx.ui["explore_person"], ctx.ui["tabs"]],
    )
