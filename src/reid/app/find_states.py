"""Messages of the Find tab: an environment that cannot be searched yet, or no result."""

from __future__ import annotations

from reid.app import components as C
from reid.app.find_data import Outcome, Selection
from reid.attributes.annotate import EnvStatus


def empty_results(out: Outcome, sel: Selection) -> str:
    res = out.result
    assert res is not None
    if res.n_in_scope == 0:
        return C.empty_state(
            "Nothing in these cameras and times", "Widen the time range or add cameras."
        )
    if sel.strict:
        return C.empty_state(
            "No one passes every filter",
            f"Turn off strict matching to see the closest candidates, or remove a filter. "
            f"{res.n_in_scope:,} appearances were checked.",
        )
    return C.empty_state("Nothing to show", "Try fewer filters.")


def state_panel(status: EnvStatus) -> str:
    """Explanation shown instead of the tool when the environment cannot be searched yet."""
    if status.state == "not_annotated":
        return C.empty_state(
            "Attributes are not computed for this environment yet",
            f"It takes a few seconds for {status.n_tracklets:,} appearances. "
            "Nothing is processed again; the stored embeddings are reused.",
        )
    if status.state == "incompatible":
        return C.empty_state(
            "Attribute search is not available here",
            "This environment was processed with a lighter body model (the lite preset) that "
            "the attribute head was not trained on. Details: "
            + status.reason.removeprefix("attributes unavailable for this environment: ")
            + ".",
        )
    if status.state == "no_head":
        return C.empty_state(
            "No attribute head installed",
            "Install one with: reid attributes use-head PATH. " + status.reason,
        )
    return C.empty_state("Nothing to search yet", "Process some footage in the Process tab.")
