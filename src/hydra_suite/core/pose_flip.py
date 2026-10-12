"""Left/right keypoint pairs for mirror augmentation.

Mirroring an image turns a left antenna into a right one, so a flip must swap
each left/right pair of keypoints too.  A skeleton stores those pairs as
``flip_pairs`` (0-based index pairs); training turns them into the
permutation Ultralytics calls ``flip_idx``.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

Pair = Tuple[int, int]

_SIDE_PATTERNS = (
    # (regex for one side, replacement giving the other side)
    (re.compile(r"left", re.IGNORECASE), "right"),
    (re.compile(r"right", re.IGNORECASE), "left"),
    (re.compile(r"(^|[_\-\s])l($|[_\-\s])", re.IGNORECASE), r"\1r\2"),
    (re.compile(r"(^|[_\-\s])r($|[_\-\s])", re.IGNORECASE), r"\1l\2"),
)


_SIDE_WORD = re.compile(r"left|right", re.IGNORECASE)
# One-letter side tokens: l / r, l1 / r2 (leg numbers), lf / rh / lm / rb
# (front, hind, middle, back).
_SIDE_TOKEN = re.compile(r"^[lr](\d+|[fhmb])?$", re.IGNORECASE)
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=\d)")


def is_side_named(name: str) -> bool:
    """Whether a keypoint name marks a left or right side.

    Deliberately broad (left/right anywhere; l/r tokens such as ``l_ear``,
    ``antL``, ``L1``, ``lf_leg``, ``l.ant``): a missed side name lets a flip
    train the model to confuse sides, a false alarm only asks for pairs.
    """
    text = str(name)
    if _SIDE_WORD.search(text):
        return True
    tokens = [t for t in _TOKEN_SPLIT.split(text) if t]
    # Re-join a letter split from its leg number ("L1" -> "L", "1").
    joined = list(tokens)
    for i in range(len(tokens) - 1):
        if tokens[i + 1].isdigit():
            joined.append(tokens[i] + tokens[i + 1])
    return any(_SIDE_TOKEN.match(t) for t in joined)


def unpaired_side_keypoints(names: Sequence[str], pairs: Iterable[Pair]) -> List[int]:
    """Side-named keypoints (see :func:`is_side_named`) that no pair covers."""
    paired = {i for pair in pairs for i in pair}
    return [i for i, n in enumerate(names) if i not in paired and is_side_named(n)]


def remap_pairs_by_name(
    pairs: Iterable[Pair], old_names: Sequence[str], new_names: Sequence[str]
) -> List[Pair]:
    """Carry pairs over to a new keypoint list by name (dropping missing ones)."""
    where = {str(n): i for i, n in enumerate(new_names)}
    out: List[Pair] = []
    for a, b in pairs:
        if not (0 <= a < len(old_names) and 0 <= b < len(old_names)):
            continue
        na, nb = where.get(str(old_names[a])), where.get(str(old_names[b]))
        if na is not None and nb is not None:
            out.append((na, nb))
    return normalize_flip_pairs(out, len(new_names))


def normalize_flip_pairs(pairs: Iterable[Sequence[int]], k: int) -> List[Pair]:
    """Valid pairs only: in range, two distinct keypoints, each used once.

    Pairs are returned as ``(low, high)`` in their original order; a pair that
    reuses a keypoint already paired is dropped.
    """
    out: List[Pair] = []
    used: set = set()
    for pair in pairs or ():
        try:
            a, b = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if a == b or not (0 <= a < k and 0 <= b < k):
            continue
        if a in used or b in used:
            continue
        used.update((a, b))
        out.append((min(a, b), max(a, b)))
    return out


def flip_index(pairs: Iterable[Sequence[int]], k: int) -> List[int]:
    """The keypoint permutation applied on a mirror (identity when no pairs)."""
    idx = list(range(k))
    for a, b in normalize_flip_pairs(pairs, k):
        idx[a], idx[b] = b, a
    return idx


def _mirror_name(name: str) -> str:
    for pattern, repl in _SIDE_PATTERNS:
        if pattern.search(name):
            swapped = pattern.sub(
                lambda m, r=repl: _match_case(m.expand(r), m.group(0)),
                name,
                count=1,
            )
            if swapped != name:
                return swapped
    return name


def _match_case(new: str, old: str) -> str:
    if old.isupper():
        return new.upper()
    if old[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def suggest_flip_pairs(names: Sequence[str]) -> List[Pair]:
    """Pairs whose names differ only by a left/right word (left_eye/right_eye,
    l_ear/r_ear, ...).  Matching ignores case."""
    lookup = {str(n).strip().lower(): i for i, n in enumerate(names)}
    pairs: List[Pair] = []
    for i, name in enumerate(names):
        other = _mirror_name(str(name).strip()).lower()
        j = lookup.get(other)
        if j is not None and j != i:
            pairs.append((i, j))
    return normalize_flip_pairs(pairs, len(names))


def remap_pairs_on_remove(pairs: Iterable[Pair], removed: int) -> List[Pair]:
    """Drop pairs that use keypoint ``removed`` and shift later indices down."""
    out: List[Pair] = []
    for a, b in pairs:
        if removed in (a, b):
            continue
        out.append((a - (a > removed), b - (b > removed)))
    return out


def remap_pairs_on_move(pairs: Iterable[Pair], old: int, new: int) -> List[Pair]:
    """Follow a keypoint moved from row ``old`` to row ``new``."""

    def remap(i: int) -> int:
        if i == old:
            return new
        if new > old and old < i <= new:
            return i - 1
        if new < old and new <= i < old:
            return i + 1
        return i

    return [(remap(a), remap(b)) for a, b in pairs]


def parse_flip_pairs(data: dict) -> List[Pair]:
    """Read ``flip_pairs`` from a skeleton / project mapping (no validation)."""
    out: List[Pair] = []
    for pair in data.get("flip_pairs") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            try:
                out.append((int(pair[0]), int(pair[1])))
            except (TypeError, ValueError):
                continue
    return out
