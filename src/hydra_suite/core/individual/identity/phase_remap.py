"""Phase-local evidence -> global catalog remapping.

Each CNN identity classifier builds its own *phase-local* cartesian catalog
from its own factors (``evidence_builder.build_phase_catalog_labels``). With
one identity model, that phase basis is identical to the global catalog and
the remap is a relabeling. With several identity models the global catalog is
the cross-product of all their axes, so a phase label such as ``"red"`` names
a *set* of global entries (``red_square``, ``red_circle``) rather than one.

This module owns that distribution. The semantics match
``substrate._factor_log_prob``'s composite branch -- assign each reachable
entry the phase probability, then renormalize -- lifted one level up from
within-model factors to across-model phases.

Core: numpy + stdlib only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.spec import IdentityCatalogSpec


def build_phase_label_map(
    spec: IdentityCatalogSpec,
    catalog: IdentityCatalog,
    model_label: str,
) -> dict[str, list[int]]:
    """``phase_display_label -> [global catalog indices]`` for one model.

    A global entry is reachable from a phase label iff the entry's classes on
    *that model's* axes, joined in axis order with "_", equal the phase label
    -- the same join ``build_phase_catalog_labels`` used to name it.
    """
    prefix = f"{model_label}:"
    out: dict[str, list[int]] = {}
    for entry in spec.entries:
        own = [cls for (axis, cls) in entry.factors if axis.startswith(prefix)]
        if not own:
            continue
        phase_label = "_".join(own)
        try:
            idx = catalog.index_of(entry.display_label)
        except KeyError:
            continue
        out.setdefault(phase_label, []).append(idx)
    return out


def remap_phase_log_probs(
    log_probs: np.ndarray,
    source_labels: Sequence[str],
    catalog: IdentityCatalog,
    phase_label_map: dict[str, list[int]],
) -> np.ndarray:
    """Map one phase-basis log-prob vector into the global catalog basis.

    Unreachable entries keep the ``1e-300`` floor rather than a hard zero, so
    the single-model case is arithmetically identical to the exact-match
    implementation this replaced.
    """
    arr = np.asarray(log_probs, dtype=np.float64)
    probs = np.exp(arr - np.max(arr))
    probs /= np.clip(probs.sum(), 1e-300, None)

    remapped = np.full(catalog.size, 1e-300, dtype=np.float64)
    for src_idx, raw_label in enumerate(source_labels):
        label = str(raw_label)
        targets = phase_label_map.get(label)
        if targets is None:
            # "unknown" (and any label with no structured provenance, e.g. a
            # tag-sourced entry) still resolves by direct lookup.
            if not catalog.contains(label):
                continue
            targets = [catalog.index_of(label)]
        for idx in targets:
            remapped[idx] += float(probs[src_idx])

    remapped /= np.clip(remapped.sum(), 1e-300, None)
    return np.log(np.clip(remapped, 1e-300, None))
