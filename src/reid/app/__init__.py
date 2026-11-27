"""Gradio app: ``build_app(cfg)`` makes the Blocks, ``launch(cfg, ...)`` serves it."""

from __future__ import annotations

import logging
import secrets
from pathlib import Path
from typing import Any

import gradio as gr

from reid.app import components as C
from reid.app import envbar
from reid.app.state import AppContext
from reid.app.tabs import admin, cameras, explore, find, metrics, overview, process, search
from reid.app.theme import ReidTheme, launch_style
from reid.config import Config

log = logging.getLogger(__name__)

__all__ = ["AppContext", "ReidTheme", "build_app", "launch", "launch_style"]

TABS = (overview, process, explore, search, find, cameras, metrics)


def build_app(
    cfg: Config,
    workspace: Path | None = None,
    env: str | None = None,
    read_only: bool = False,
) -> gr.Blocks:
    """Create the app. No model is loaded and no run is started until a tab asks for it.

    ``workspace`` is the folder whose subfolders are the environments and ``env`` the one
    selected first. Without them ``cfg.data_dir`` is the initial environment and its parent
    folder the workspace. ``read_only`` leaves out the delete and wipe tools.

    Gradio 6 takes theme and CSS at launch time: call :func:`launch`, or pass
    ``**launch_style()`` to ``Blocks.launch``.
    """
    ctx = AppContext(cfg, read_only=read_only, workspace=workspace, env=env)
    with gr.Blocks(title="reid", analytics_enabled=False) as demo:
        ctx.ui["demo"] = demo
        gr.HTML(
            C.page_header(
                "reid",
                "Who walked where? Multi-camera person re-identification with face confirmation.",
                [C.badge("read-only", "warn")] if read_only else [],
            )
        )
        envbar.build(ctx)
        with gr.Tabs() as tabs:
            ctx.ui["tabs"] = tabs
            for module in TABS:
                module.build(ctx)
            admin.build(ctx)
    demo.ctx = ctx  # type: ignore[attr-defined]
    return demo


def launch(
    cfg: Config,
    share: bool = False,
    auth: tuple[str, str] | None = None,
    workspace: Path | None = None,
    env: str | None = None,
    read_only: bool = False,
    port: int | None = None,
    host: str | None = None,
    block: bool = True,
) -> tuple[gr.Blocks, tuple[str, str] | None]:
    """Serve the app and return ``(demo, credentials)``.

    Sharing a public link without a password is refused: with ``share=True`` and no ``auth`` a
    password is generated and returned (and logged). ``read_only`` hides the delete and wipe
    tools.
    """
    if share and auth is None:
        auth = ("reid", secrets.token_urlsafe(9))
        log.warning("share=True without auth: generated login %s / %s", *auth)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    demo = build_app(cfg, workspace=workspace, env=env, read_only=read_only)
    kwargs: dict[str, Any] = {
        **launch_style(),
        "allowed_paths": [str(demo.ctx.workspace)],  # type: ignore[attr-defined]
        "share": share,
        "auth": auth,
        "server_port": port,
        "server_name": host,
        "prevent_thread_lock": not block,
        "show_error": True,
        "footer_links": [],
    }
    demo.queue().launch(**kwargs)
    return demo, auth
