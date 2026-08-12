"""Content signature for the identity-evidence sidecar cache.

Independent of the raw CNN/AprilTag cache keys: it folds in the catalog domain
and the per-factor calibration temperatures so a recalibration or catalog change
invalidates only the evidence sidecar, never the raw caches.
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

from hydra_suite.core.individual.identity.spec import IdentityCatalogSpec


def identity_evidence_cache_key(
    catalog_spec: IdentityCatalogSpec,
    per_factor_temps: Mapping[str, tuple[float, ...]],
    base_signature: str,
) -> str:
    payload = {
        "catalog": catalog_spec.to_dict(),
        "temps": {
            k: [round(float(t), 6) for t in v]
            for k, v in sorted(per_factor_temps.items())
        },
        "base": str(base_signature),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]
