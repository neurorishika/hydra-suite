"""Label Review: browse every image of a label, likely mislabels first.

Pure numpy, no Qt.  :func:`label_probabilities` scores how much the model
agrees with each assigned label (``P(label | image)``), :func:`predicted_labels`
gives the model's own guess, and :class:`LabelReviewModel` orders a label's
images by that agreement and tracks removals with undo.

Removing a label only clears it: the image goes back to unlabeled.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_SEP = "_"  # MULTIHEAD_LABEL_SEPARATOR

SOURCES = ("all", "human", "machine")


def _composite_names(factor_labels: Sequence[Sequence[str]]) -> Dict[str, tuple]:
    names: Dict[str, tuple] = {}
    for combo in product(*[[str(v) for v in f] for f in factor_labels]):
        name = _SEP.join(combo)
        if name in names:
            raise ValueError(
                f"Combination name {name!r} is ambiguous; factor values must not "
                f"contain {_SEP!r}."
            )
        names[name] = combo
    return names


def _check(factor_probs, factor_labels, n: int) -> List[np.ndarray]:
    if len(factor_probs) != len(factor_labels):
        raise ValueError("factor_probs and factor_labels differ in length")
    out = []
    for fp, fl in zip(factor_probs, factor_labels):
        arr = np.asarray(fp, dtype=np.float64)
        if arr.ndim != 2 or arr.shape != (n, len(fl)):
            raise ValueError(
                f"factor probabilities of shape {arr.shape} do not match "
                f"{n} images x {len(fl)} values"
            )
        out.append(arr)
    return out


def label_probabilities(
    labels: Sequence[Optional[str]],
    factor_probs: Sequence[np.ndarray],
    factor_labels: Sequence[Sequence[str]],
    joint_probs: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """``P(assigned label | image)`` per image.

    Uses the joint column when given (flat models), else the product of
    per-factor marginals.  NaN for unlabeled images and labels the model
    cannot score (not in the scheme).
    """
    n = len(labels)
    probs = _check(factor_probs, factor_labels, n)
    names = _composite_names(factor_labels)
    joint = {
        str(k): np.asarray(v, dtype=np.float64) for k, v in (joint_probs or {}).items()
    }
    out = np.full(n, np.nan)
    groups: Dict[str, List[int]] = {}
    for i, lab in enumerate(labels):
        if lab:
            groups.setdefault(str(lab), []).append(i)
    for lab, idx in groups.items():
        rows = np.asarray(idx)
        if lab in joint and joint[lab].shape == (n,):
            p = joint[lab][rows]
        elif lab in names:
            p = np.ones(rows.size)
            for f, value in enumerate(names[lab]):
                col = [str(v) for v in factor_labels[f]].index(value)
                p = p * probs[f][rows, col]
        else:
            continue
        out[rows] = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def predicted_labels(
    factor_probs: Sequence[np.ndarray],
    factor_labels: Sequence[Sequence[str]],
    joint_probs: Optional[Dict[str, np.ndarray]] = None,
) -> List[Optional[str]]:
    """The model's top label per image (``None`` where it has no finite output)."""
    if joint_probs:
        names = [str(k) for k in joint_probs]
        cols = [np.asarray(joint_probs[k], dtype=np.float64) for k in joint_probs]
        n = cols[0].shape[0]
        mat = np.stack(cols, axis=1)
        finite = np.isfinite(mat).any(axis=1)
        best = np.argmax(np.where(np.isfinite(mat), mat, -np.inf), axis=1)
        return [names[best[i]] if finite[i] else None for i in range(n)]
    if not factor_probs:
        return []
    n = np.asarray(factor_probs[0]).shape[0]
    probs = _check(factor_probs, factor_labels, n)
    parts = []
    finite = np.ones(n, dtype=bool)
    for fp, fl in zip(probs, factor_labels):
        ok = np.isfinite(fp).any(axis=1)
        finite &= ok
        best = np.argmax(np.where(np.isfinite(fp), fp, -np.inf), axis=1)
        parts.append([str(fl[b]) for b in best])
    return [_SEP.join(p[i] for p in parts) if finite[i] else None for i in range(n)]


@dataclass
class Removal:
    """One undoable removal: ``(index, label)`` pairs cleared together."""

    entries: List[Tuple[int, str]]


class LabelReviewModel:
    """Every labeled image grouped by label, likely mislabels first."""

    def __init__(
        self,
        labels: Sequence[Optional[str]],
        machine_mask: Optional[Sequence[bool]] = None,
        agreement: Optional[np.ndarray] = None,
        predicted: Optional[Sequence[Optional[str]]] = None,
    ) -> None:
        self.n = len(labels)
        self.labels: List[Optional[str]] = [str(lab) if lab else None for lab in labels]
        self.machine = (
            np.zeros(self.n, dtype=bool)
            if machine_mask is None
            else np.asarray(machine_mask, dtype=bool)
        )
        self.agreement = None if agreement is None else np.asarray(agreement, float)
        self.predicted = None if predicted is None else list(predicted)
        for name, arr in (
            ("machine_mask", self.machine),
            ("agreement", self.agreement),
            ("predicted", self.predicted),
        ):
            if arr is not None and len(arr) != self.n:
                raise ValueError(f"{name} has {len(arr)} entries for {self.n} labels")
        self.history: List[Removal] = []

    @property
    def has_model(self) -> bool:
        return self.agreement is not None or self.predicted is not None

    def _source_ok(self, i: int, source: str) -> bool:
        if source == "human":
            return not self.machine[i]
        if source == "machine":
            return bool(self.machine[i])
        return True

    def disagrees(self, i: int) -> bool:
        """The model's top label differs from the assigned one."""
        if self.predicted is None or not self.labels[i]:
            return False
        pred = self.predicted[i]
        return pred is not None and pred != self.labels[i]

    def label_names(self, source: str = "all") -> List[str]:
        return sorted(
            {
                lab
                for i, lab in enumerate(self.labels)
                if lab and self._source_ok(i, source)
            }
        )

    def summary(self, source: str = "all") -> Dict[str, Tuple[int, int]]:
        """``{label: (count, disagreements)}`` for the given source filter."""
        out: Dict[str, List[int]] = {}
        for i, lab in enumerate(self.labels):
            if lab and self._source_ok(i, source):
                entry = out.setdefault(lab, [0, 0])
                entry[0] += 1
                entry[1] += int(self.disagrees(i))
        return {k: (v[0], v[1]) for k, v in sorted(out.items())}

    def items(
        self, label: str, source: str = "all", disagree_only: bool = False
    ) -> List[int]:
        """Images labeled ``label``: lowest model agreement first, then index."""
        idx = [
            i
            for i, lab in enumerate(self.labels)
            if lab == label
            and self._source_ok(i, source)
            and (not disagree_only or self.disagrees(i))
        ]

        def key(i: int):
            a = None if self.agreement is None else self.agreement[i]
            if a is None or not np.isfinite(a):
                return (1, 0.0, i)
            return (0, float(a), i)

        return sorted(idx, key=key)

    def remove(self, indices: Sequence[int]) -> Optional[Removal]:
        """Clear the labels of ``indices`` as one undoable step."""
        entries = []
        for i in dict.fromkeys(int(i) for i in indices):
            if 0 <= i < self.n and self.labels[i]:
                entries.append((i, self.labels[i]))
                self.labels[i] = None
        if not entries:
            return None
        removal = Removal(entries)
        self.history.append(removal)
        return removal

    def undo(self) -> Optional[Removal]:
        """Restore the most recent removal."""
        if not self.history:
            return None
        removal = self.history.pop()
        for i, lab in removal.entries:
            self.labels[i] = lab
        return removal

    def removed_count(self) -> int:
        return sum(len(r.entries) for r in self.history)
