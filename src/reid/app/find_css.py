"""Stylesheet rules of the Find tab that depend on the filter catalogue.

Gradio renders every choice of a ``CheckboxGroup`` as ``<label><input name=VALUE>``, so a rule such
as ``label:has(input[name="upper_red"])`` can give one chip its own look: the swatch colours and
the dashed outline of the less reliable filters come from ``reid.attributes.ui_labels`` instead of
being written twice.
"""

from __future__ import annotations

from reid.attributes import ui_labels


def catalogue_css() -> str:
    """Swatch colours (``--sw``) and the ``rd-soft`` look of unreliable chips."""
    rules = []
    for item in ui_labels.UI_LABELS.values():
        sel = f'.rd-filter label:has(input[name="{item.id}"])'
        if item.hex:
            rules.append(f"{sel} {{ --sw: {item.hex}; }}")
        if not item.reliable:
            rules.append(f"{sel} {{ border-style: dashed !important; }}")
    return "\n".join(rules)
