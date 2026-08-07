"""Identity catalog — the ordered domain of known identities for one run.

Identity Phase 0: all evidence sources (AprilTag, CNN) map into this index space.
The catalog is immutable after construction and is passed through to all
online decoding and post-processing components.

Index 0 is always the ``unknown`` / unobserved state.
Indices 1..N correspond to the configured unique-identity labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

UNKNOWN_LABEL: str = "unknown"
"""Reserved label for the unobserved / ambiguous state (always at index 0)."""


@dataclass(frozen=True)
class IdentityCatalog:
    """Ordered identity domain for one tracking run.

    ``labels[0]`` is always ``UNKNOWN_LABEL``.  Indices 1..N correspond to the
    configured unique-identity labels in the order they were registered.

    The catalog is hashable and safe to use as a dict key or in frozen sets.

    Parameters
    ----------
    labels:
        Full label tuple including the leading ``UNKNOWN_LABEL``.  Use
        :meth:`from_labels` to construct from a plain list of known labels.
    """

    labels: tuple[str, ...]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def from_labels(known_labels: Sequence[str]) -> "IdentityCatalog":
        """Create a catalog from a sequence of known (non-unknown) labels.

        The unknown state is prepended automatically at index 0.

        Raises
        ------
        ValueError
            If ``UNKNOWN_LABEL`` appears in *known_labels* (it is reserved).
        ValueError
            If *known_labels* is empty.
        """
        if not known_labels:
            raise ValueError("known_labels must not be empty")
        known = list(known_labels)
        if UNKNOWN_LABEL in known:
            raise ValueError(
                f"'{UNKNOWN_LABEL}' must not appear in the known labels list; "
                "it is reserved as the unobserved state at index 0."
            )
        return IdentityCatalog(labels=(UNKNOWN_LABEL, *known))

    # ------------------------------------------------------------------
    # Size / membership
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total catalog size including the unknown slot."""
        return len(self.labels)

    @property
    def num_known(self) -> int:
        """Number of known (non-unknown) identities."""
        return len(self.labels) - 1

    @property
    def unknown_index(self) -> int:
        """Always 0."""
        return 0

    def index_of(self, label: str) -> int:
        """Return the catalog index for *label*.

        Raises
        ------
        KeyError
            If *label* is not in the catalog.
        """
        try:
            return self.labels.index(label)
        except ValueError:
            raise KeyError(label) from None

    def label_of(self, index: int) -> str:
        """Return the label for *index*.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        return self.labels[index]

    def is_unknown(self, index: int) -> bool:
        """True if *index* corresponds to the unknown/unobserved state."""
        return index == self.unknown_index

    def contains(self, label: str) -> bool:
        """True if *label* is in the catalog (including unknown)."""
        return label in self.labels

    def known_indices(self) -> range:
        """Indices of all known (non-unknown) identities: range(1, size)."""
        return range(1, self.size)

    # ------------------------------------------------------------------
    # Log-prior helpers
    # ------------------------------------------------------------------

    def uniform_log_prior(self) -> np.ndarray:
        """Flat log-prior over the full catalog.

        Returns shape ``(size,)`` float64 with the unknown slot having equal
        weight to each known identity.  This is the weakest possible prior.
        """
        C = self.size
        return np.full(C, -np.log(C), dtype=np.float64)

    def known_uniform_log_prior(self, unknown_weight: float = 0.05) -> np.ndarray:
        """Log-prior with a small total mass on the unknown state.

        Parameters
        ----------
        unknown_weight:
            Total probability mass allocated to the unknown slot.
            The remaining mass is spread uniformly across known identities.

        Returns shape ``(size,)`` float64.
        """
        known_p = (1.0 - unknown_weight) / max(self.num_known, 1)
        priors = np.full(self.size, known_p, dtype=np.float64)
        priors[0] = unknown_weight
        return np.log(np.clip(priors, 1e-300, None))

    def apriltag_log_prior(
        self,
        tag_id: int,
        tag_to_label: dict[int, str],
        floor: float = 1e-4,
    ) -> np.ndarray:
        """Sharp categorical log-prior for an AprilTag observation.

        Almost all the mass is placed on the identity mapped from *tag_id*.
        A small floor probability is spread across all other identities so the
        posterior is never driven to ``-inf`` on a single conflicting
        observation.

        Parameters
        ----------
        tag_id:
            Observed AprilTag integer ID.
        tag_to_label:
            Mapping from AprilTag ID to catalog label.
        floor:
            Probability floor for non-matching identities.

        Returns shape ``(size,)`` float64.
        """
        C = self.size
        n_other = max(C - 1, 1)
        p = np.full(C, floor / n_other, dtype=np.float64)
        p[0] = floor  # unknown also gets floor

        label = tag_to_label.get(tag_id)
        if label is not None and self.contains(label):
            idx = self.index_of(label)
            p[idx] = 1.0 - floor

        # Renormalise (small numerical correction)
        p /= p.sum()
        return np.log(np.clip(p, 1e-300, None))

    def cnn_log_prior(
        self,
        class_probs: np.ndarray,
        label_map: list[str],
        floor: float = 1e-6,
    ) -> np.ndarray:
        """Map CNN output probabilities to a catalog log-prior.

        Parameters
        ----------
        class_probs:
            Shape ``(K,)`` raw CNN softmax / probability output, one value per
            class in the CNN model's output space.
        label_map:
            Length-K list mapping CNN output index to catalog label.
        floor:
            Probability floor for catalog entries not covered by the CNN.

        Returns shape ``(size,)`` float64.
        """
        p = np.full(self.size, floor, dtype=np.float64)
        p[0] = floor  # unknown gets the floor

        for k, label in enumerate(label_map):
            if self.contains(label) and label != UNKNOWN_LABEL:
                idx = self.index_of(label)
                p[idx] = max(float(class_probs[k]), floor)

        p /= p.sum()
        return np.log(np.clip(p, 1e-300, None))

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"IdentityCatalog(size={self.size}, "
            f"labels={self.labels[:min(5, self.size)]}{'...' if self.size > 5 else ''})"
        )
