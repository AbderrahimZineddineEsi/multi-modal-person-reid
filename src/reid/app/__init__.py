"""Gradio app: ``build_app(cfg)`` makes the Blocks, ``launch(cfg, ...)`` serves it."""

from __future__ import annotations

import logging
import secrets
from typing import Any

import gradio as gr

from reid.app import components as C
from reid.app.state import AppContext
from reid.app.tabs import admin, cameras, explore, metrics, overview, process, search
from reid.app.theme import ReidTheme, launch_style
from reid.config import Config

log = logging.getLogger(__name__)

__all__ = ["AppContext", "ReidTheme", "build_app", "launch", "launch_style"]

TABS = (overview, process, explore, search, cameras, metrics)


def build_app(cfg: Config, admin_tools: bool = False) -> gr.Blocks:
    """Create the app. No model is loaded and no run is started until a tab asks for it.

    Gradio 6 takes theme and CSS at launch time: call :func:`launch`, or pass
    ``**launch_style()`` to ``Blocks.launch``.
    """
    ctx = AppContext(cfg, admin=admin_tools)
    with gr.Blocks(title="reid", analytics_enabled=False) as demo:
        ctx.ui["demo"] = demo
        gr.HTML(
            C.page_header(
                "reid",
                "Who walked where? Multi-camera person re-identification with face confirmation.",
                [C.badge(cfg.data_dir.name or str(cfg.data_dir), "dim")]
                + ([C.badge("admin mode", "warn")] if admin_tools else []),
            )
        )
        with gr.Tabs() as tabs:
            ctx.ui["tabs"] = tabs
            for module in TABS:
                module.build(ctx)
            if admin_tools:
                admin.build(ctx)
    demo.ctx = ctx  # type: ignore[attr-defined]
    return demo


def launch(
    cfg: Config,
    share: bool = False,
    auth: tuple[str, str] | None = None,
    admin: bool = False,
    port: int | None = None,
    host: str | None = None,
    block: bool = True,
) -> tuple[gr.Blocks, tuple[str, str] | None]:
    """Serve the app and return ``(demo, credentials)``.

    Sharing a public link without a password is refused: with ``share=True`` and no ``auth`` a
    password is generated and returned (and logged). Destructive admin tools exist only with
    ``admin=True``.
    """
    if share and auth is None:
        auth = ("reid", secrets.token_urlsafe(9))
        log.warning("share=True without auth: generated login %s / %s", *auth)
    if share and admin:
        log.warning("admin tools are enabled on a shared link")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    demo = build_app(cfg, admin_tools=admin)
    kwargs: dict[str, Any] = {
        **launch_style(),
        "allowed_paths": [str(demo.ctx.data_dir)],  # type: ignore[attr-defined]
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
