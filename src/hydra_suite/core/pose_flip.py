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

# Ways a keypoint name marks its side.  Each rule matches the side marker
# (group "s"); swapping it gives the partner's name.  Strong markers are
# unambiguous; a weak one (a lone l/r glued to a word: lhip, rear) counts only
# when the mirrored name exists too ("rear" alone is not a side).
# A side letter ends its token: end of name, a separator or digit, or the
# start of a capitalised word (LFoot).  So LEG, RAY1, BRAIN are not sides.
_END = r"(?=$|[^A-Za-z]|[A-Z][a-z])"
_SIDE_RULES = (
    # left / right as a word or word start (left_eye, leftwing; not "bright")
    (re.compile(r"(?<![a-z])(?P<s>[Ll]eft|LEFT|[Rr]ight|RIGHT)"), True),
    (re.compile(r"(?<=[a-z0-9])(?P<s>Left|Right)"), True),  # eyeLeft
    # a lone l/r: l_ear, ear_r, L1, l.ant, LFoot
    (re.compile(r"(?<![A-Za-z])(?P<s>[lLrR])" + _END), True),
    # camelCase suffix: antL, eyeR, legL1
    (re.compile(r"(?<=[a-z0-9])(?P<s>[LR])" + _END), True),
    # l/r + front/hind/middle/back: lf_leg, RH, lfpaw (weak: BL, MR can be
    # other things)
    (
        re.compile(
            r"(?<![A-Za-z])(?P<s>[lLrR])(?=[fFhHmMbB](?:$|[^A-Za-z]|[A-Z][a-z]|[a-z]{2,}))"
        ),
        False,
    ),
    # front/hind/middle/back + l/r: FL, hr_paw, paw_fl (weak)
    (re.compile(r"(?<![A-Za-z])[fFhHmMbB](?P<s>[lLrR])" + _END), False),
    # l/r glued to a word: lhip, Leye, lantenna (weak)
    (re.compile(r"(?<![A-Za-z])(?P<s>[lLrR])(?=[a-z]{2,})"), False),
)
_SWAP = {"left": "right", "right": "left", "l": "r", "r": "l"}


def _swap_side(marker: str) -> str:
    other = _SWAP[marker.lower()]
    if marker.isupper():
        return other.upper()
    if marker[:1].isupper():
        return other[:1].upper() + other[1:]
    return other


def _side_variants(name: str) -> List[Tuple[str, bool]]:
    """``(partner name, strong)`` for each side marker found in ``name``."""
    out: List[Tuple[str, bool]] = []
    for pattern, strong in _SIDE_RULES:
        for m in pattern.finditer(name):
            a, b = m.span("s")
            out.append((name[:a] + _swap_side(m.group("s")) + name[b:], strong))
    return out


def is_side_named(name: str, names: Sequence[str] = ()) -> bool:
    """Whether a keypoint name marks a left or right side.

    Broad on purpose (left_eye, l_ear, antL, L1, lf_leg, FL, l.ant; lhip when
    rhip is among ``names``): a missed side name lets a flip train the model to
    confuse sides, a false alarm only asks for pairs.
    """
    known = {str(n).strip().lower() for n in names}
    for partner, strong in _side_variants(str(name).strip()):
        if strong or partner.lower() in known:
            return True
    return False


def unpaired_side_keypoints(names: Sequence[str], pairs: Iterable[Pair]) -> List[int]:
    """Side-named keypoints (see :func:`is_side_named`) that no pair covers."""
    paired = {i for pair in pairs for i in pair}
    return [
        i for i, n in enumerate(names) if i not in paired and is_side_named(n, names)
    ]


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


def suggest_flip_pairs(names: Sequence[str]) -> List[Pair]:
    """Pairs whose names differ only by their side marker (left_eye/right_eye,
    l_ear/r_ear, antL/antR, L1/R1, FL/FR, lhip/rhip, ...).  Ignores case."""
    lookup = {str(n).strip().lower(): i for i, n in enumerate(names)}
    pairs: List[Pair] = []
    for i, name in enumerate(names):
        for partner, _strong in _side_variants(str(name).strip()):
            j = lookup.get(partner.lower())
            if j is not None and j != i:
                pairs.append((i, j))
                break
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
