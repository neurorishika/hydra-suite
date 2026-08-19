"""Identity-head resolution: which CNN classifiers may influence identity.

A *identity head* is a ``CNN_CLASSIFIERS`` entry with
``unique_identifier=True``. Only identity heads may feed the identity
catalog, the identity evidence summary columns, or ``UniqueIdentityKey``.
Classifiers that are not identity heads (behavior, sex, caste) keep their
own ``CNN_<label>_*`` output columns and influence nothing about identity.

This module is Core: stdlib only, no numpy/pandas/Qt, no app-layer imports.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


class _HeadsUnknown:
    """Sentinel: no classifier config available, use legacy all-columns behavior."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "HEADS_UNKNOWN"


HEADS_UNKNOWN = _HeadsUnknown()


def identity_head_labels(cnn_classifiers: Sequence[Mapping]) -> tuple[str, ...]:
    """Labels of the classifiers that are identity providers, in config order."""
    labels: list[str] = []
    for cfg in cnn_classifiers or []:
        if not bool(cfg.get("unique_identifier", False)):
            continue
        label = str(cfg.get("label", "") or "").strip()
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


def resolve_identity_heads(params: Mapping):
    """Identity-head labels from engine params, or ``HEADS_UNKNOWN``.

    ``HEADS_UNKNOWN`` (no ``CNN_CLASSIFIERS`` key at all) means the caller is
    running over a dataframe with no engine config -- e.g. re-running
    post-processing on a bare CSV -- and must fall back to the legacy
    "every CNN column counts" behavior. A *present but empty* list, or a
    list with no identity provider in it, is a real answer: no CNN column
    feeds identity.
    """
    if "CNN_CLASSIFIERS" not in params:
        return HEADS_UNKNOWN
    return identity_head_labels(params.get("CNN_CLASSIFIERS") or [])


def identity_class_columns(
    columns: Iterable, head_labels: Sequence[str], all_labels=()
) -> list[str]:
    """The ``CNN_<head>[_<factor>]_Class`` columns belonging to identity heads.

    Matched against the *known* head labels rather than by regex capture:
    ``^CNN_(.+)_Class$`` cannot distinguish a head label containing "_"
    from a ``<label>_<factor>`` pair, and guessing wrong silently drops or
    admits the wrong classifier.

    When ``all_labels`` is supplied (identity + non-identity classifier roster),
    uses longest-match to disambiguate: a column is an identity column only if
    the longest label in ``all_labels | head_labels`` that underscore-boundary-
    prefixes it is itself an identity head. When ``all_labels`` is empty (the
    default), uses the legacy behavior for byte-identity with equivalence gates.
    """
    heads = [str(h) for h in head_labels if str(h).strip()]
    out: list[str] = []

    # If all_labels not provided, use legacy behavior
    if not all_labels:
        for col in columns:
            name = str(col)
            if not name.endswith("_Class"):
                continue
            for head in heads:
                if name == f"CNN_{head}_Class" or name.startswith(f"CNN_{head}_"):
                    out.append(name)
                    break
        return out

    # New behavior: longest-match against full roster
    all_labels_list = [str(l) for l in all_labels if str(l).strip()]
    all_possible_labels = list(set(heads) | set(all_labels_list))

    for col in columns:
        name = str(col)
        if not name.endswith("_Class"):
            continue

        # Find longest label that matches at underscore boundary
        longest_match = None
        for label in all_possible_labels:
            # Match 1: exact CNN_<label>_Class
            if name == f"CNN_{label}_Class":
                if longest_match is None or len(label) > len(longest_match):
                    longest_match = label
            # Match 2: CNN_<label>_<something> where <something> is not empty
            elif name.startswith(f"CNN_{label}_"):
                if longest_match is None or len(label) > len(longest_match):
                    longest_match = label

        # Include if longest match is an identity head
        if longest_match and longest_match in heads:
            out.append(name)

    return out
