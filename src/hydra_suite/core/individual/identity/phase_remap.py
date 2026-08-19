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

import logging
from typing import Mapping, Sequence

import numpy as np

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.spec import IdentityCatalogSpec

logger = logging.getLogger(__name__)


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


def phase_map_key(cfg: Mapping) -> str:
    """The ``source_name`` a classifier's evidence is looked up by.

    Must match ``worker.py``'s ``_cnn_phase_states`` label expression
    (``str(cfg.get("label", "cnn_identity"))``) exactly -- that is the
    ``source_name`` stamped on every ``IdentityEvidence`` this classifier
    emits, and therefore the key both the live path and the offline path
    look its phase map up by.
    """
    return str(cfg.get("label", "cnn_identity"))


def phase_axis_model_label(cfg: Mapping) -> str:
    """The axis prefix a classifier's catalog entries were built with.

    Must match ``resolve.identity_axes()``'s ``model_label`` normalization
    (``str(cfg.get("label", "") or "").strip() or "cnn"``). It differs from
    :func:`phase_map_key` for a whitespace-padded or absent label, and the
    two must be paired up or the map silently misses.
    """
    return str(cfg.get("label", "") or "").strip() or "cnn"


def build_phase_label_maps(
    spec: IdentityCatalogSpec,
    catalog: IdentityCatalog,
    cnn_classifiers: Sequence[Mapping],
) -> dict[str, dict[str, list[int]]]:
    """``source_name -> phase label map`` for every identity classifier.

    The one build-time owner of the phase-map construction *and* of its
    diagnostics, shared by the live tracking worker and the offline fragment
    solver so the two cannot drift. Three config-time warnings are raised
    here (never on a per-detection/per-row path):

    - an **empty** map: this classifier reaches zero catalog entries, so all
      its evidence would floor;
    - a **zero-overlap** map: the map is non-empty but none of its keys is
      one of this classifier's own phase labels, so every lookup misses and
      the evidence floors just as completely -- the failure mode an
      empty-map check alone cannot see;
    - **colliding** model labels: two identity classifiers whose labels
      normalize to the same axis prefix, which makes both models' axes join
      into one map and guarantees the zero-overlap failure.
    """
    from hydra_suite.core.individual.identity.evidence_builder import (
        build_phase_catalog_labels,
    )

    maps: dict[str, dict[str, list[int]]] = {}
    seen_axis_labels: dict[str, str] = {}
    for cfg in cnn_classifiers or []:
        if not bool(cfg.get("unique_identifier", False)):
            continue
        map_key = phase_map_key(cfg)
        axis_label = phase_axis_model_label(cfg)

        if axis_label in seen_axis_labels:
            logger.warning(
                "Identity classifiers %r and %r both normalize to the axis "
                "prefix '%s'. Their axes are indistinguishable in the "
                "identity catalog, so at least one classifier's evidence "
                "will be entirely floored. Give each identity classifier a "
                "distinct 'label'.",
                seen_axis_labels[axis_label],
                map_key,
                axis_label,
            )
        else:
            seen_axis_labels[axis_label] = map_key

        pmap = build_phase_label_map(spec, catalog, axis_label)
        if not pmap:
            logger.warning(
                "Identity classifier '%s' (unique_identifier=True) "
                "maps to ZERO entries in the resolved identity "
                "catalog -- its evidence will be entirely floored "
                "and effectively discarded. Check this classifier's "
                "'label' and 'class_names_per_factor' configuration "
                "against the other configured identity classifiers.",
                map_key,
            )
        else:
            phase_labels = set(
                build_phase_catalog_labels(
                    [list(f or []) for f in (cfg.get("class_names_per_factor") or [])]
                )
            )
            phase_labels.discard("unknown")
            if phase_labels and not (phase_labels & set(pmap)):
                logger.warning(
                    "Identity classifier '%s' (unique_identifier=True) has a "
                    "non-empty phase map whose keys (%s) share NOTHING with "
                    "the phase labels its evidence is actually written "
                    "against (%s) -- every lookup will miss and its evidence "
                    "will be entirely floored. Check this classifier's "
                    "'label'/'factor_names'/'class_names_per_factor' against "
                    "the other configured identity classifiers.",
                    map_key,
                    ", ".join(sorted(pmap)),
                    ", ".join(sorted(phase_labels)),
                )
        maps[map_key] = pmap
    return maps
