"""Identity association: self-adaptive galleries, constraints, UDM, consolidation, prototypes.

``reid.identity.consolidate`` is a submodule (its ``consolidate`` function is deliberately not
re-exported, so the name does not shadow the module).
"""

from reid.identity.associate import Decision, IdentityEngine, RunIdentityReport
from reid.identity.consolidate import (
    ConsolidationResult,
    merge_identities,
    recluster,
    rename_identity,
    split_tracklet,
)
from reid.identity.prototypes import rebuild_prototypes

__all__ = [
    "ConsolidationResult",
    "Decision",
    "IdentityEngine",
    "RunIdentityReport",
    "merge_identities",
    "rebuild_prototypes",
    "recluster",
    "rename_identity",
    "split_tracklet",
]
