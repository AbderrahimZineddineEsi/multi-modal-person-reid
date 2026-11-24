"""Environment bar shown above the tabs: pick, create and refresh environments.

Switching re-points the shared :class:`AppContext` and bumps ``ctx.ui["env_version"]``. Every
tab lists that state's ``change`` event next to ``tab.select`` and ``demo.load`` as a trigger,
so what is displayed follows the environment (see :func:`refresh_on`).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import gradio as gr

from reid.app import components as C
from reid.app import environments as envs
from reid.app.state import AppContext


def refresh_on(ctx: AppContext, *triggers: Any) -> list[Any]:
    """``[*triggers, env changed]``: the events after which a tab reloads its main outputs."""
    return [*triggers, ctx.ui["env_version"].change]


def choices(ctx: AppContext) -> list[tuple[str, str]]:
    return [(e.label, e.name) for e in ctx.environments()]


def dropdown_update(ctx: AppContext):
    """Dropdown with fresh stats, the current environment selected."""
    return gr.update(choices=choices(ctx), value=ctx.env)


def status_html(ctx: AppContext, text: str = "") -> str:
    """Small note next to the bar (empty unless something was created or deleted)."""
    return C.badge(text, "accent") if text else ""


def bump() -> float:
    """New value for the version state: any change fires the tabs' refresh."""
    return time.time()


def build(ctx: AppContext) -> gr.Row:
    """Create the bar and register ``ctx.ui["env_dropdown" | "env_version"]``."""
    ctx.ui["env_version"] = gr.State(0.0)
    with gr.Row(equal_height=True, elem_classes="rd-envbar") as bar:
        env = gr.Dropdown(
            choices(ctx), value=ctx.env, label="Environment", scale=5, filterable=False,
            container=False,
        )  # fmt: skip
        new_btn = gr.Button("New environment", size="sm", scale=0, min_width=150)
        refresh_btn = gr.Button("Refresh", size="sm", scale=0, min_width=80)
        msg = gr.HTML(elem_classes="rd-envmsg")
    with gr.Row(visible=False, elem_classes="rd-envbar") as new_row:
        name = gr.Textbox(label="Name", placeholder="shop_floor_1", scale=3, max_lines=1,
                          info="Letters, digits, - and _.")  # fmt: skip
        desc = gr.Textbox(label="Description (optional)", scale=3, max_lines=1)
        create_btn = gr.Button("Create and switch", variant="primary", scale=0, min_width=160)
    ctx.ui["env_dropdown"] = env
    ctx.ui["env_message"] = msg
    version = ctx.ui["env_version"]

    def switch(target: str | None):
        if not target:
            return gr.update(), gr.update(), ""
        try:
            ctx.select_env(target)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return dropdown_update(ctx), bump(), status_html(ctx)

    def create(new_name: str, description: str):
        try:
            envs.create_environment(ctx.workspace, new_name, description or "")
            ctx.select_env(new_name.strip())
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return (
            dropdown_update(ctx), bump(), status_html(ctx, f"created {ctx.env}"),
            gr.update(visible=False), "",
        )  # fmt: skip

    def refresh():
        return dropdown_update(ctx), bump(), status_html(ctx)

    outs = [env, version, msg]
    env.input(switch, env, outs)
    refresh_btn.click(refresh, None, outs)
    new_btn.click(lambda: gr.update(visible=True), None, new_row)
    create_btn.click(create, [name, desc], [*outs, new_row, name])
    ctx.ui["demo"].load(
        lambda: (dropdown_update(ctx), gr.update(), status_html(ctx)), None, outs
    )  # a new browser tab shows the current environment; the tabs load their own data
    return bar


def refreshed(ctx: AppContext, message: str = "") -> Sequence[Any]:
    """Values for ``[env_dropdown, env_version, env_message]`` after a change made elsewhere."""
    return dropdown_update(ctx), bump(), message or status_html(ctx)
