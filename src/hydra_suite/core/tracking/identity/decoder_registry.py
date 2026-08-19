"""One OnlineIdentityDecoder per arena, behind the single-decoder call surface.

Identity labels repeat per arena -- arena 1 and arena 2 may each contain an
"antA" -- so the decoder's one-individual-one-track uniqueness constraint must
be scoped per arena. ``OnlineIdentityDecoder`` is already self-contained and
slot-keyed, so scoping needs no change inside it: one instance per arena over
the shared catalog is exactly the required semantic ("one antA in *this*
arena").

This registry exposes the same methods ``worker.py`` already calls on a bare
decoder -- ``get_belief``, ``clear_slot``, ``decay_absent_slot_beliefs``,
``get_slot_log_posteriors``, ``all_active_slots``, ``update_frame``, plus the
``._catalog`` attribute worker.py reads directly -- routing each by
``slot_arena[slot]``. With a single arena the registry holds exactly one
decoder, so every call reaches that decoder with the same arguments a bare
decoder would receive today.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.online import (
    IdentityAssignment,
    OnlineIdentityDecoder,
    TrackIdentityBelief,
)


class ArenaDecoderRegistry:
    """Slot-routed collection of per-arena ``OnlineIdentityDecoder`` instances.

    Slots are partitioned by ``slot_arena`` (never renumbered); each arena's
    decoder enforces identity uniqueness only over the slots belonging to it,
    so the same catalog label may be assigned independently in each arena.
    """

    def __init__(
        self,
        catalog: IdentityCatalog,
        params: dict[str, Any],
        slot_arena: np.ndarray,
    ) -> None:
        # Arena-invariant: the same catalog object is shared by every arena's
        # decoder. Exposed as ``._catalog`` so worker.py's direct attribute
        # reads (e.g. ``_identity_online_decoder._catalog``) keep working
        # unchanged against the registry.
        self._catalog = catalog
        self._params = params
        self.slot_arena = np.asarray(slot_arena, dtype=np.int32)
        if self.slot_arena.ndim != 1 or self.slot_arena.size == 0:
            raise ValueError(
                "slot_arena must be a non-empty 1-D array; got shape "
                f"{self.slot_arena.shape}"
            )
        if np.any(self.slot_arena < 0):
            raise ValueError(
                "slot_arena must not contain negative arena ids; got "
                f"{self.slot_arena.tolist()}"
            )
        self.decoders: dict[int, OnlineIdentityDecoder] = {
            int(arena_id): OnlineIdentityDecoder(catalog, params)
            for arena_id in np.unique(self.slot_arena)
        }

    @property
    def n_decoders(self) -> int:
        return len(self.decoders)

    def _arena_of(self, slot_index: int) -> int:
        """Look up the arena id for *slot_index*, bounds-checked.

        A bare ``slot_arena[slot_index]`` would silently wrap on a negative
        index and raise an opaque ``IndexError`` when out of range -- the
        latter is dangerous because at least one worker.py call site
        (``get_slot_log_posteriors`` inside the Bayesian identity cost term,
        ``worker.py:2782``) sits inside a bare ``except Exception`` that would
        swallow it and silently disable the cost term. Fail loudly and
        explicitly here instead.
        """
        idx = int(slot_index)
        if idx < 0 or idx >= len(self.slot_arena):
            raise IndexError(
                f"slot_index {idx} out of bounds for slot_arena of length "
                f"{len(self.slot_arena)}"
            )
        return int(self.slot_arena[idx])

    def decoder_for_slot(self, slot_index: int) -> OnlineIdentityDecoder:
        """Return the decoder owning *slot_index*'s arena."""
        return self.decoders[self._arena_of(slot_index)]

    def _group_slots_by_arena(self, slots: Iterable[int]) -> dict[int, list[int]]:
        grouped: dict[int, list[int]] = {}
        for slot in slots:
            grouped.setdefault(self._arena_of(slot), []).append(int(slot))
        return grouped

    # ------------------------------------------------------------------
    # Single-decoder call surface (matches OnlineIdentityDecoder exactly)
    # ------------------------------------------------------------------

    def get_belief(self, slot_index: int) -> TrackIdentityBelief | None:
        return self.decoder_for_slot(slot_index).get_belief(slot_index)

    def clear_slot(
        self,
        slot_index: int,
        reason: str = "",
        respawn_frame_idx: int | None = None,
    ) -> None:
        self.decoder_for_slot(slot_index).clear_slot(
            slot_index, reason=reason, respawn_frame_idx=respawn_frame_idx
        )

    def decay_absent_slot_beliefs(self, absent_slots: list[int]) -> None:
        for arena, slots in self._group_slots_by_arena(absent_slots).items():
            self.decoders[arena].decay_absent_slot_beliefs(slots)

    def get_slot_log_posteriors(self, slots: list[int]) -> dict[int, np.ndarray]:
        merged: dict[int, np.ndarray] = {}
        for arena, arena_slots in self._group_slots_by_arena(slots).items():
            merged.update(self.decoders[arena].get_slot_log_posteriors(arena_slots))
        return merged

    def all_active_slots(self) -> list[int]:
        """Return active slot indices, concatenated across arena decoders.

        Order: per-decoder insertion order (as ``OnlineIdentityDecoder.
        all_active_slots`` returns, per ``online.py``), concatenated in
        ``self.decoders`` iteration order (ascending arena id). This
        deliberately does *not* sort -- a bare decoder's
        ``all_active_slots()`` returns insertion order, not sorted order, and
        there is no existing caller relying on a sorted result.
        """
        return [
            slot
            for decoder in self.decoders.values()
            for slot in decoder.all_active_slots()
        ]

    def update_frame(
        self,
        frame_idx: int,
        visible_slots: list[int],
        slot_evidences: dict[int, list],
    ) -> list[IdentityAssignment]:
        """Run each arena's decoder over only that arena's visible slots.

        ``visible_slots`` and ``slot_evidences`` are partitioned by
        ``slot_arena`` and handed to each arena's decoder unchanged (same
        argument shapes a bare decoder receives today), so uniqueness is
        enforced per arena. Results are concatenated -- track slot indices
        are globally unique, so there is nothing to merge/dedupe across
        arenas, unlike ``get_slot_log_posteriors`` which returns a dict.

        Contract change vs. a bare decoder: ``OnlineIdentityDecoder.
        update_frame`` documents its return as "one assignment per visible
        slot, in the same order as visible_slots" (``online.py:402``). This
        registry instead returns assignments grouped arena-by-arena (each
        arena's block in that arena's visible-slot order), which need not
        match the original ``visible_slots`` order when slots interleave
        across arenas. Harmless for every current caller -- ``worker.py``
        immediately folds the result into a ``dict`` keyed by
        ``slot_index`` -- but future callers must not assume order.
        """
        assignments: list[IdentityAssignment] = []
        for arena, arena_slots in self._group_slots_by_arena(visible_slots).items():
            arena_evidences = {
                slot: slot_evidences[slot]
                for slot in arena_slots
                if slot in slot_evidences
            }
            assignments.extend(
                self.decoders[arena].update_frame(
                    frame_idx, arena_slots, arena_evidences
                )
            )
        return assignments
