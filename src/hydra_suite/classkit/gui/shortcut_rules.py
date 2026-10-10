"""Shortcut collision rules shared by the shortcut editor and scheme editor.

Two kinds of keys live side by side: global actions (approve, reject, next,
undo, ...) and per-label scheme keys.  A key bound to both makes Qt fire
neither (or the wrong one), so the editors refuse such bindings up front.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

from PySide6.QtGui import QKeySequence

# Keys the app binds itself, independent of the editable global actions.
BUILTIN_KEYS: Dict[str, str] = {"0": "assign 'unknown'"}

# Defaults shipped before approve/reject moved to +/-; a saved copy of these is
# treated as "never customised" so users pick up the new keys.
LEGACY_DEFAULTS: Dict[str, str] = {
    "Approve review label": "Y",
    "Reject review label": "N",
}


def normalize_key(seq: object) -> str:
    """Canonical comparison form of a key (``""`` for none)."""
    text = str(seq or "").strip()
    if not text:
        return ""
    return QKeySequence(text).toString(QKeySequence.PortableText).upper() or text.upper()


def global_key_map(shortcuts: Mapping[str, str]) -> Dict[str, str]:
    """``{normalized key: action name}`` for the global actions."""
    out: Dict[str, str] = {}
    for action, seq in shortcuts.items():
        key = normalize_key(seq)
        if key and key not in out:
            out[key] = action
    return out


def reserved_key_map(shortcuts: Mapping[str, str]) -> Dict[str, str]:
    """Everything a scheme label key must not use."""
    reserved = {normalize_key(k): v for k, v in BUILTIN_KEYS.items()}
    reserved.update(global_key_map(shortcuts))
    return reserved


def scheme_key_conflicts(
    factors: Iterable[Mapping], reserved: Mapping[str, str]
) -> List[str]:
    """Problems with the label keys of ``factors`` (``name``/``labels``/``shortcuts``).

    Within one factor each key may be used once (the same key in *different*
    factors is fine: factors are labelled one after another).  No key may equal a
    reserved/global one.
    """
    problems: List[str] = []
    for factor in factors:
        fname = str(factor.get("name", "factor"))
        labels = list(factor.get("labels", []))
        keys = list(factor.get("shortcuts", factor.get("shortcut_keys", [])))
        seen: Dict[str, str] = {}
        for label, raw in zip(labels, keys):
            key = normalize_key(raw)
            if not key:
                continue
            if key in reserved:
                problems.append(
                    f"'{raw}' on {fname}/{label} is already used by "
                    f"the global shortcut: {reserved[key]}"
                )
            elif key in seen:
                problems.append(
                    f"'{raw}' is assigned to both {fname}/{seen[key]} and "
                    f"{fname}/{label}"
                )
            else:
                seen[key] = label
    return problems


def global_shortcut_conflicts(
    shortcuts: Mapping[str, str], scheme_keys: Mapping[str, str]
) -> List[str]:
    """Problems with a set of global action bindings.

    ``scheme_keys`` is ``{normalized key: "factor/label"}`` for the current scheme.
    """
    problems: List[str] = []
    seen: Dict[str, str] = {}
    builtin = {normalize_key(k): v for k, v in BUILTIN_KEYS.items()}
    for action, raw in shortcuts.items():
        key = normalize_key(raw)
        if not key:
            continue
        if key in seen:
            problems.append(f"'{raw}' is bound to both '{seen[key]}' and '{action}'")
        else:
            seen[key] = action
        if key in scheme_keys:
            problems.append(
                f"'{raw}' ({action}) is already the label key for {scheme_keys[key]}"
            )
        if key in builtin:
            problems.append(f"'{raw}' ({action}) is reserved to {builtin[key]}")
    return problems


def scheme_key_map(factors: Iterable[Mapping]) -> Dict[str, str]:
    """``{normalized key: "factor/label"}`` over every factor of a scheme."""
    out: Dict[str, str] = {}
    for factor in factors:
        fname = str(factor.get("name", "factor"))
        keys = factor.get("shortcuts", factor.get("shortcut_keys", []))
        for label, raw in zip(factor.get("labels", []), keys):
            key = normalize_key(raw)
            if key and key not in out:
                out[key] = f"{fname}/{label}"
    return out
