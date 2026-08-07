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
import os
from typing import Mapping, Sequence

from hydra_suite.core.individual.identity.spec import CatalogEntry, IdentityCatalogSpec


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

    for cfg in cnn_classifiers or []:
        if not bool(cfg.get("unique_identifier", False)):
            continue
        cnpf = list(cfg.get("class_names_per_factor") or [])
        if not cnpf:
            cnpf = _read_factors_from_model_file(str(cfg.get("model_path", "")))
        factor_names = list(cfg.get("factor_names") or [])
        non_empty = [fl for fl in cnpf if fl]

        if len(non_empty) > 1:
            for combo in itertools.product(*non_empty):
                pairs = tuple(
                    (factor_names[i] if i < len(factor_names) else f"factor{i}", str(c))
                    for i, c in enumerate(combo)
                    if c
                )
                display = "_".join(str(c) for c in combo if c)
                _add(display, pairs, "cnn")
        else:
            flat: list[str] = []
            for fl in non_empty:
                flat.extend([str(x) for x in fl if x])
            if not flat:
                flat = [str(x) for x in (cfg.get("labels", []) or []) if x]
            fname = factor_names[0] if factor_names else "factor0"
            for lbl in flat:
                _add(lbl, ((fname, lbl),), "cnn")

    cnn_derived = set(seen)
    for lbl in tag_identity_labels or []:
        s = str(lbl).strip() if lbl else ""
        if not s:
            continue
        if cnn_derived and s not in cnn_derived:
            continue
        _add(s, (), "tag")

    return IdentityCatalogSpec(entries=tuple(entries))
