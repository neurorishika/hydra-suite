"""CAPTCHA-style grid balancing over label combinations.

The human sees one combination at a time (``a_x``, then ``b_y``, ...) and a
grid of images that might belong to it, and clicks every tile that does.  A
clicked tile is labeled with that combination and immediately replaced; an
unclicked tile gets **no** label -- it could be anything other than this one
combination -- and is simply never shown again for that combination.

Combinations are visited rarest first, and each one is filled up to the count
of the most common combination (optionally capped).  Tiles mix three sources so
the grid is quick to answer but still informative and keeps the labeler alert:

* ``high``    -- the most likely remaining members (fast, easy hits);
* ``mid``     -- drawn from the next band down (borderline cases);
* ``explore`` -- uniform over everything else (catches what the model misses).

Probabilities come from per-factor marginals (see :mod:`.combination`), so a
combination the model has never output can still be targeted.  Pure numpy (no
Qt), so the whole session is unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence

import numpy as np

from .combination import count_labeled_combinations

_UNKNOWN = "unknown"
_SEP = "_"
SOURCES = ("high", "mid", "explore")


@dataclass
class Hit:
    """One committed click, kept for undo."""

    index: int
    combo: str
    slot: int
    replacement: Optional[int]


@dataclass
class ClickResult:
    index: int
    label: str
    completed: bool  # this click brought the combination to its target


class GridBalanceSession:
    """State of one balancing session; the dialog is a thin view over it.

    Args:
        factor_probs: per-factor marginals ``[(N, L_f)]``.
        factor_labels: the scheme's labels per factor (one list for a flat scheme).
        image_labels: current label per image; only images with no label are
            ever shown (a pending machine label is left for review instead).
        trusted_mask: True where a label counts toward the balance (human
            verified).  ``None`` trusts every label.
        grid_size: tiles per grid.
        include_unknown: also balance combinations containing ``unknown``.
        target_cap: upper bound on the per-combination target.
        fractions: share of ``high`` / ``mid`` / ``explore`` tiles.
        joint_probs: optional ``{combo: (N,)}`` probabilities used instead of
            the product of marginals (a flat composite model already outputs
            the joint, including correlations between factors).
    """

    def __init__(
        self,
        factor_probs: Sequence[np.ndarray],
        factor_labels: Sequence[Sequence[str]],
        image_labels: Sequence[Optional[str]],
        trusted_mask: Optional[np.ndarray] = None,
        *,
        grid_size: int = 9,
        include_unknown: bool = False,
        target_cap: Optional[int] = None,
        fractions: Sequence[float] = (0.5, 0.375, 0.125),
        seed: int = 0,
        joint_probs: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        if len(factor_probs) != len(factor_labels) or not factor_labels:
            raise ValueError("factor_probs and factor_labels must align")
        self.factor_labels = [[str(v) for v in f] for f in factor_labels]
        self._probs = [np.asarray(p, dtype=np.float64) for p in factor_probs]
        self.n = int(self._probs[0].shape[0])
        if len(image_labels) != self.n:
            raise ValueError("image_labels length does not match predictions")
        self.grid_size = max(1, int(grid_size))
        frac = np.clip(np.asarray(fractions, dtype=np.float64), 0.0, None)
        self._fractions = frac / frac.sum() if frac.sum() > 0 else np.ones(3) / 3
        self._rng = np.random.default_rng(seed)

        self.labels: List[Optional[str]] = [lab or None for lab in image_labels]
        self._initially_labeled = np.array(
            [bool(lab) for lab in self.labels], dtype=bool
        )
        self._unlabeled = ~self._initially_labeled
        self._trusted = (
            np.ones(self.n, dtype=bool)
            if trusted_mask is None
            else np.asarray(trusted_mask, dtype=bool)
        )

        combos = list(product(*self.factor_labels))
        if not include_unknown:
            combos = [c for c in combos if _UNKNOWN not in c]
        raw = count_labeled_combinations(self.labels, self.factor_labels, self._trusted)
        names = [_SEP.join(c) for c in combos]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Factor values must not contain '{_SEP}': combination names collide"
            )
        self._joint = {
            str(k): np.asarray(v, dtype=np.float64)
            for k, v in (joint_probs or {}).items()
            if np.asarray(v).shape == (self.n,)
        }
        self._p: Dict[str, np.ndarray] = {}
        self._combo_parts = {name: c for name, c in zip(names, combos)}
        self.counts: Dict[str, int] = {
            name: int(raw.get(c, 0)) for name, c in zip(names, combos)
        }
        top = max(self.counts.values(), default=0)
        if top <= 0:
            top = int(target_cap) if target_cap else self.grid_size
        self.target = min(top, int(target_cap)) if target_cap else top
        # Rarest first; combinations already at target are not visited.
        self.combos: List[str] = sorted(
            (c for c in names if self.counts[c] < self.target),
            key=lambda c: (self.counts[c], c),
        )

        self._order: Dict[str, np.ndarray] = {}
        self._shown: Dict[str, np.ndarray] = {}
        self.hits: List[Hit] = []
        self.pos = 0
        self.grid: List[Optional[int]] = []
        self.sources: List[Optional[str]] = []
        if self.combos:
            self._enter(0)

    # ── queries ──────────────────────────────────────────────────────────
    @property
    def current(self) -> Optional[str]:
        return self.combos[self.pos] if 0 <= self.pos < len(self.combos) else None

    @property
    def finished(self) -> bool:
        return self.current is None

    def parts(self, combo: str) -> tuple:
        """Per-factor values of ``combo``."""
        return self._combo_parts[combo]

    def probability(self, combo: str) -> np.ndarray:
        """``P(combo | image)``: the joint column if given, else the product of
        per-factor marginals (which can target never-predicted combinations)."""
        cached = self._p.get(combo)
        if cached is not None:
            return cached
        if combo in self._joint:
            out = self._joint[combo].copy()
        else:
            out = np.ones(self.n, dtype=np.float64)
            for f, value in enumerate(self._combo_parts[combo]):
                out *= self._probs[f][:, self.factor_labels[f].index(value)]
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        self._p[combo] = out
        return out

    def remaining(self, combo: Optional[str] = None) -> int:
        """Unshown, unlabeled images still available for ``combo``."""
        combo = combo or self.current
        if combo is None:
            return 0
        return int(self._eligible(combo).size)

    def examples(self, combo: Optional[str] = None, k: int = 3) -> List[int]:
        """Up to ``k`` trusted labeled examples of ``combo`` (reference tiles)."""
        combo = combo or self.current
        out = []
        for i, lab in enumerate(self.labels):
            if lab == combo and (self._trusted[i] or not self._initially_labeled[i]):
                out.append(i)
                if len(out) >= k:
                    break
        return out

    def session_labeled(self) -> int:
        return len(self.hits)

    # ── actions ──────────────────────────────────────────────────────────
    def click(self, slot: int) -> Optional[ClickResult]:
        """Label the tile in ``slot`` as the current combination and refill it."""
        combo = self.current
        if combo is None or not 0 <= slot < len(self.grid):
            return None
        idx = self.grid[slot]
        if idx is None:
            return None
        self.labels[idx] = combo
        self._unlabeled[idx] = False
        self.counts[combo] += 1
        repl, src = self._draw(combo, self._pick_source())
        self.grid[slot], self.sources[slot] = repl, src
        self.hits.append(Hit(idx, combo, slot, repl))
        return ClickResult(idx, combo, self.counts[combo] >= self.target)

    def refresh(self) -> None:
        """None match: every visible tile stays shown, a new grid is drawn."""
        if self.current is not None:
            self._fill_grid(self.current)

    def replace(self, slot: int) -> Optional[int]:
        """Swap the tile in ``slot`` for a new candidate without labeling it
        (e.g. its image could not be read)."""
        combo = self.current
        if combo is None or not 0 <= slot < len(self.grid):
            return None
        self.grid[slot], self.sources[slot] = self._draw(combo, self._pick_source())
        return self.grid[slot]

    def exclude(self, index: int) -> None:
        """Never show ``index`` again in any combination (e.g. unreadable image).

        It stays unlabeled; it simply leaves every candidate pool.
        """
        if 0 <= index < self.n and self.labels[index] is None:
            self._unlabeled[index] = False

    def resize(self, grid_size: int) -> None:
        """Change the grid size; the visible tiles go back into the pool."""
        combo = self.current
        self._release_grid()
        self.grid_size = max(1, int(grid_size))
        if combo is not None:
            self._fill_grid(combo)

    def next_combo(self) -> bool:
        """Move to the next combination still below target; False when done."""
        for p in range(self.pos + 1, len(self.combos)):
            if self.counts[self.combos[p]] < self.target:
                self._enter(p)
                return True
        self.pos = len(self.combos)
        self.grid, self.sources = [], []
        return False

    def undo(self) -> Optional[Hit]:
        """Revert the last click and put the image back where it was."""
        if not self.hits:
            return None
        hit = self.hits.pop()
        self.labels[hit.index] = None
        self._unlabeled[hit.index] = True
        self.counts[hit.combo] -= 1
        if hit.combo != self.current:
            self._release_grid()  # unanswered tiles stay available
            self._enter(self.combos.index(hit.combo))
        slot = min(hit.slot, len(self.grid) - 1)
        displaced = self.grid[slot]
        if displaced is not None and displaced != hit.index:
            self._shown[hit.combo][displaced] = False  # back into the pool
        self.grid[slot], self.sources[slot] = hit.index, "undo"
        return hit

    # ── internals ────────────────────────────────────────────────────────
    def _release_grid(self) -> None:
        """Return the visible, unanswered tiles to the current combination's pool."""
        combo = self.current
        if combo is None:
            return
        for i in self.grid:
            if i is not None:
                self._shown[combo][i] = False

    def _enter(self, pos: int) -> None:
        self.pos = pos
        combo = self.combos[pos]
        self._ensure(combo)
        self._fill_grid(combo)

    def _ensure(self, combo: str) -> None:
        if combo not in self._order:
            # Stable sort so equal probabilities keep image order.
            self._order[combo] = np.argsort(-self.probability(combo), kind="stable")
            self._shown[combo] = np.zeros(self.n, dtype=bool)

    def _eligible(self, combo: str) -> np.ndarray:
        self._ensure(combo)
        order = self._order[combo]
        ok = self._unlabeled & ~self._shown[combo]
        for i in self.grid if combo == self.current else ():
            if i is not None:
                ok[i] = False
        return order[ok[order]]

    def _pick_source(self) -> str:
        return SOURCES[int(self._rng.choice(3, p=self._fractions))]

    def _fill_grid(self, combo: str) -> None:
        k = self.grid_size
        n_explore = int(round(k * self._fractions[2])) if k >= 4 else 0
        if self._fractions[2] > 0 and k >= 4:
            n_explore = max(1, n_explore)
        n_high = int(round(k * self._fractions[0]))
        if self._fractions[0] > 0:
            n_high = max(1, n_high)  # the top candidate is always on the grid
        n_mid = max(0, k - n_high - n_explore)
        plan = ["high"] * n_high + ["mid"] * n_mid + ["explore"] * n_explore
        plan = plan[:k]
        self.grid = [None] * k
        self.sources = [None] * k
        for slot, src in zip(self._rng.permutation(k), plan):
            self.grid[slot], self.sources[slot] = self._draw(combo, src)

    def _draw(self, combo: str, source: str):
        """Take one image for ``combo`` from ``source``; marks it shown."""
        elig = self._eligible(combo)
        if elig.size == 0:
            return None, None
        lo, hi = self._mid_band(combo, elig)
        if source == "high" or elig.size == 1:
            idx = int(elig[0])
        elif source == "mid" and hi > lo:
            idx = int(elig[self._rng.integers(lo, hi)])
        elif source == "explore" and elig.size > hi:
            idx = int(elig[self._rng.integers(hi, elig.size)])
        else:
            idx = int(elig[self._rng.integers(0, elig.size)])
        self._shown[combo][idx] = True
        return idx, source

    def _mid_band(self, combo: str, elig: np.ndarray):
        """Rank window ``[lo, hi)`` of borderline candidates in ``elig``.

        Sized by probability, not rank count, so it stays "lower but not
        unlikely" on large datasets: it ends at about twice the expected number
        of remaining members, and never goes below a tenth of the current top
        candidate's probability.
        """
        p = self.probability(combo)[elig]
        lo = min(elig.size, self.grid_size)
        mass_end = lo + int(np.ceil(2.0 * float(p.sum())))
        plausible_end = int(np.searchsorted(-p, -0.1 * float(p[0]), side="right"))
        hi = min(elig.size, mass_end, plausible_end)
        return lo, max(hi, min(elig.size, lo + 1))
