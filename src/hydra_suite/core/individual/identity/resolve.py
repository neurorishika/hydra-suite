"""The single identity-catalog resolver.

Given the configured CNN classifiers and tag labels, produce the ordered,
structured :class:`IdentityCatalogSpec`. This is the one place the identity
domain is assembled; the tracking worker consumes it via
``IdentityCatalog.from_spec``. Ported behavior-for-behavior from the former
inline assembly in ``core/tracking/worker.py`` (composite display labels are
"_"-joined exactly as before), with the addition of structured factor keys.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
from typing import Mapping, NamedTuple, Sequence

from hydra_suite.core.individual.identity.spec import CatalogEntry, IdentityCatalogSpec

logger = logging.getLogger(__name__)

CATALOG_SIZE_WARN_THRESHOLD = 256
"""Entry count above which the cross-product catalog is flagged as suspicious.

The Hungarian assignment cost matrix is N x (K + N) in the catalog size K, so
a runaway product is a real cost. This is a warning, not a cap: naming the
contributing axes is more useful than refusing to run.
"""


class IdentityAxis(NamedTuple):
    """One factor axis of the identity domain.

    ``model_label`` is the owning classifier's label, ``factor_name`` its
    factor's name (positional ``factor<i>`` fallback when the config carries
    no ``factor_names``), and ``classes`` that factor's class vocabulary.
    """

    model_label: str
    factor_name: str
    classes: tuple[str, ...]

    @property
    def qualified_name(self) -> str:
        return f"{self.model_label}:{self.factor_name}"


def identity_axes(cnn_classifiers: Sequence[Mapping]) -> list[IdentityAxis]:
    """The identity domain's factor axes, in model-config then factor order.

    Every identity-providing model contributes each of its non-empty factors
    as one axis. The catalog is the cartesian product over ALL axes -- two
    complementary tag models (thorax colour + abdomen shape) describe one
    animal jointly, so their classes multiply rather than compete.

    Redundant voters (two models over the same class vocabulary) are not
    supported and would produce nonsensical ``ant1_ant1`` composites; see the
    Non-goals section of the design doc.
    """
    axes: list[IdentityAxis] = []
    for cfg in cnn_classifiers or []:
        if not bool(cfg.get("unique_identifier", False)):
            continue
        model_label = str(cfg.get("label", "") or "").strip() or "cnn"
        cnpf = list(cfg.get("class_names_per_factor") or [])
        if not cnpf:
            cnpf = _read_factors_from_model_file(str(cfg.get("model_path", "")))
        factor_names = list(cfg.get("factor_names") or [])
        non_empty_index = 0
        for i, factor_labels in enumerate(cnpf):
            classes = tuple(str(c) for c in (factor_labels or []) if c)
            if not classes:
                continue
            name = (
                str(factor_names[i])
                if i < len(factor_names) and str(factor_names[i]).strip()
                else f"factor{non_empty_index}"
            )
            non_empty_index += 1
            axes.append(IdentityAxis(model_label, name, classes))
    return axes


def _read_factors_from_model_file(model_path: str) -> list[list[str]]:
    """Fallback: read per-factor class names from a model manifest JSON.

    Mirrors worker.py:1852-1872 — prefer ``class_names_per_factor``, then a
    flat ``class_names``, then ``factor_models[].class_names``.
    """
    try:
        if not model_path or not os.path.exists(model_path):
            return []
        with open(model_path) as fh:
            meta = json.load(fh)
    except Exception:
        return []
    cnpf = meta.get("class_names_per_factor") or []
    if cnpf:
        return cnpf
    flat = meta.get("class_names") or []
    if flat:
        return [flat]
    out: list[list[str]] = []
    for fe in meta.get("factor_models") or []:
        fl = fe.get("class_names") or []
        if fl:
            out.append(fl)
    return out


def non_identifying_marks(
    cnn_classifiers: Sequence[Mapping],
) -> dict[str, tuple[str, ...]]:
    """``model_label -> declared non-identifying marks`` for identity models."""
    out: dict[str, tuple[str, ...]] = {}
    for cfg in cnn_classifiers or []:
        if not bool(cfg.get("unique_identifier", False)):
            continue
        label = str(cfg.get("label", "") or "").strip() or "cnn"
        marks = tuple(
            str(m).strip()
            for m in (cfg.get("non_identifying_classes") or [])
            if str(m).strip()
        )
        if marks:
            out[label] = marks
    return out


def is_non_identifying(
    combo: Sequence[str],
    axes: Sequence[IdentityAxis],
    marks_by_model: Mapping[str, Sequence[str]],
    display_label: str,
) -> bool:
    """True if this composite is excluded by any declared mark.

    Three accepted mark forms, all resolved here so the rest of the system
    only ever sees per-entry exclusion:

    - ``"notag"``       -- that class in any axis of the declaring model
    - ``"front:notag"`` -- that class in the named factor of that model
    - ``"notag_notag"`` -- that whole global composite display label
    """
    for model_label, marks in marks_by_model.items():
        for mark in marks:
            if mark == display_label:
                return True
            scoped_factor, _, scoped_class = mark.partition(":")
            for i, axis in enumerate(axes):
                if axis.model_label != model_label or i >= len(combo):
                    continue
                cls = str(combo[i])
                if scoped_class:
                    if axis.factor_name == scoped_factor and cls == scoped_class:
                        return True
                elif cls == mark:
                    return True
    return False


def excluded_display_labels(
    cnn_classifiers: Sequence[Mapping],
) -> frozenset[str]:
    """Every composite display label the declared marks exclude.

    The reporting layer uses this to recognize an observed composite as
    non-identifying without re-deriving the mark semantics.
    """
    axes = identity_axes(cnn_classifiers)
    marks = non_identifying_marks(cnn_classifiers)
    if not axes or not marks:
        return frozenset()
    out = set()
    for combo in itertools.product(*[a.classes for a in axes]):
        display = "_".join(str(c) for c in combo if c)
        if display and is_non_identifying(combo, axes, marks, display):
            out.add(display)
    return frozenset(out)


def resolve_catalog_spec(
    cnn_classifiers: Sequence[Mapping[str, object]],
    tag_identity_labels: Sequence[object],
) -> IdentityCatalogSpec:
    """Resolve the ordered identity domain from CNN + tag configuration."""
    entries: list[CatalogEntry] = []
    seen: set[str] = set()

    def _add(display: str, factors: tuple[tuple[str, str], ...], source: str) -> None:
        if display and display not in seen:
            seen.add(display)
            entries.append(
                CatalogEntry(display_label=display, factors=factors, source=source)
            )

    axes = identity_axes(cnn_classifiers)
    marks_by_model = non_identifying_marks(cnn_classifiers)
    if axes:
        projected = 1
        for a in axes:
            projected *= max(1, len(a.classes))
        if projected > CATALOG_SIZE_WARN_THRESHOLD:
            logger.warning(
                "Identity catalog is the cross-product of %d axes and has %d "
                "entries (> %d). Contributing axes: %s. Check that every "
                "classifier marked 'unique identifier' really is one, and "
                "consider marking non-identifying classes.",
                len(axes),
                projected,
                CATALOG_SIZE_WARN_THRESHOLD,
                ", ".join(f"{a.qualified_name}({len(a.classes)})" for a in axes),
            )
        for combo in itertools.product(*[a.classes for a in axes]):
            display = "_".join(str(c) for c in combo if c)
            if marks_by_model and is_non_identifying(
                combo, axes, marks_by_model, display
            ):
                # Excluded from the identity domain entirely: no Hungarian
                # column, no blocked_labels entry, no commit-blocking, no swap
                # candidacy, no offline collision veto. Any number of tracks
                # may carry this class, and none is ever merged onto another
                # because of it.
                continue
            pairs = tuple(
                (axes[i].qualified_name, str(c)) for i, c in enumerate(combo) if c
            )
            _add(display, pairs, "cnn")

        if marks_by_model and not seen:
            marks_summary = ", ".join(
                f"{m}: {', '.join(v)}" for m, v in sorted(marks_by_model.items())
            )
            logger.warning(
                "Every identity in the catalog was excluded by the declared "
                "non-identifying classes (%s). Identity resolution will not "
                "run for this session." % marks_summary
            )

    cnn_derived = set(seen)
    for lbl in tag_identity_labels or []:
        s = str(lbl).strip() if lbl else ""
        if not s:
            continue
        if cnn_derived and s not in cnn_derived:
            continue
        _add(s, (), "tag")

    return IdentityCatalogSpec(entries=tuple(entries))
