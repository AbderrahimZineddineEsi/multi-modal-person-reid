"""Render ``configs/default.yaml`` (every parameter commented) from the config models."""

from __future__ import annotations

import json
import textwrap
from typing import Any

from pydantic import BaseModel

from reid.config import Config

HEADER = """\
# reid default configuration.
# Generated from src/reid/config.py (python -m reid.config_yaml > configs/default.yaml).
# Every key is optional in your own file: missing keys fall back to these defaults.
# Each parameter below is documented as: meaning (default).
"""


def _scalar(v: Any) -> str:
    return json.dumps(v)  # JSON is valid YAML flow syntax (null, true, [1, 2], "str")


def render_default_yaml() -> str:
    """Return the full commented default YAML text."""
    lines = [HEADER]
    cfg = Config()
    for sname in Config.model_fields:
        section: BaseModel = getattr(cfg, sname)
        doc = (type(section).__doc__ or "").strip()
        lines.append(f"# --- {sname}: {doc}")
        lines.append(f"{sname}:")
        for fname, finfo in type(section).model_fields.items():
            value = getattr(section, fname)
            text = f"{finfo.description} (default: {_scalar(value)})"
            for chunk in textwrap.wrap(text, 92):
                lines.append(f"  # {chunk}")
            lines.append(f"  {fname}: {_scalar(value)}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_default_yaml(), end="")
