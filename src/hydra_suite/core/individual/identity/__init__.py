"""Identity resolution: catalog, evidence, calibration, cache, and the
realtime (``online``) and post-hoc (``offline``) consumers.

Slotted here by the identity-overhaul Phase 0 directory reorganization
(behavior-preserving move; see
``docs/superpowers/specs/2026-07-22-identity-overhaul-consolidated-design.md``).
"""

from hydra_suite.core.individual.identity.catalog import UNKNOWN_LABEL, IdentityCatalog
from hydra_suite.core.individual.identity.spec import CatalogEntry, IdentityCatalogSpec

__all__ = ["IdentityCatalog", "UNKNOWN_LABEL", "CatalogEntry", "IdentityCatalogSpec"]
