"""Track editor data model for RefineKit.

Provides a *fragment*-based editing model that powers the timeline editor.
Every trajectory in the visible region is split into one or more
:class:`TrackFragment` objects.  The user manipulates fragments via three
atomic operations:

* **Split** — cut a fragment at a specific frame, producing two fragments.
* **Delete** — remove a fragment entirely.
* **Reassign** — move a fragment to a different track lane (only if the
  target lane has no overlapping fragment).

All edits are staged in-memory; nothing touches disk until
:meth:`TrackEditorModel.apply` is called, which emits a list of
:class:`EditOp` objects consumable by :mod:`correction_writer`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Edit operation descriptors  (what gets sent to the writer)
# ---------------------------------------------------------------------------


class OpKind(enum.Enum):
    """Kind of edit operation produced by :meth:`TrackEditorModel.apply`."""

    DELETE = "delete"
    REASSIGN = "reassign"


@dataclass
class EditOp:
    """A single concrete edit that the correction writer can execute.

    Attributes
    ----------
    kind:
        The operation type.
    track_id:
        The *original* track ID this operation acts on.
    frame_start, frame_end:
        Inclusive frame range of the fragment.
    new_track_id:
        For REASSIGN: the target track ID.
    """

    kind: OpKind
    track_id: int
    frame_start: int
    frame_end: int
    new_track_id: Optional[int] = None


# ---------------------------------------------------------------------------
# TrackFragment
# ---------------------------------------------------------------------------


@dataclass
class TrackFragment:
    """A contiguous segment of a single trajectory.

    Attributes
    ----------
    frag_id:
        Unique monotonically increasing identifier (stable across edits).
    track_id:
        The lane this fragment belongs to (may change via reassign).
    original_track_id:
        The track ID this fragment had when the editor was opened.
    frame_start, frame_end:
        Inclusive frame range.
    deleted:
        Whether the user marked this fragment for deletion.
    """

    frag_id: int
    track_id: int
    original_track_id: int
    frame_start: int
    frame_end: int
    deleted: bool = False


# ---------------------------------------------------------------------------
# TrackEditorModel
# ---------------------------------------------------------------------------


class TrackEditorModel:
    """In-memory fragment model for timeline-based track editing.

    Parameters
    ----------
    df:
        The full trajectory DataFrame.
    visible_tracks:
        Track IDs to show in the editor.
    frame_range:
        ``(start, end)`` inclusive frame range visible in the editor.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        visible_tracks: List[int],
        frame_range: Tuple[int, int],
    ) -> None:
        self._full_frame_range = frame_range
        self._view_range = frame_range
        self._starting_tracks = sorted({int(track_id) for track_id in visible_tracks})
        all_track_ids = self._starting_tracks + [
            int(track_id)
            for track_id in df.get("TrajectoryID", pd.Series(dtype=int))
            .dropna()
            .tolist()
        ]
        self._next_track_id = max(all_track_ids or [0]) + 1
        self._next_frag_id = 0
        self._fragments: List[TrackFragment] = []
        self._extra_tracks: set[int] = set()
        self._history: List[tuple[List[TrackFragment], set[int]]] = []  # undo stack

        self._build_fragments(df)
        self._extra_tracks = set(self._starting_tracks) - self._occupied_track_ids()

    # ------------------------------------------------------------------
    # Fragment construction
    # ------------------------------------------------------------------

    def _build_fragments(self, df: pd.DataFrame) -> None:
        """Convert DataFrame rows into full contiguous fragments.

        Fragment boundaries follow the underlying trajectory continuity, not
        the editor's loaded window. Any contiguous segment that overlaps the
        loaded frame range is included so deletes/reassigns apply to the full
        segment the user is editing.
        """
        fstart, fend = self._full_frame_range
        sub = df[df["TrajectoryID"].isin(self._starting_tracks)].copy()

        for tid, grp in sub.groupby("TrajectoryID"):
            frames = sorted(grp["FrameID"].unique())
            if not frames:
                continue
            # Find contiguous runs
            seg_start = frames[0]
            prev = frames[0]
            for f in frames[1:]:
                if f > prev + 1:
                    if seg_start <= fend and prev >= fstart:
                        self._add_fragment(int(tid), seg_start, prev)
                    seg_start = f
                prev = f
            if seg_start <= fend and prev >= fstart:
                self._add_fragment(int(tid), seg_start, prev)

    def _add_fragment(self, track_id: int, fs: int, fe: int) -> TrackFragment:
        frag = TrackFragment(
            frag_id=self._next_frag_id,
            track_id=track_id,
            original_track_id=track_id,
            frame_start=fs,
            frame_end=fe,
        )
        self._next_frag_id += 1
        self._fragments.append(frag)
        return frag

    # ------------------------------------------------------------------
    # Snapshot / undo
    # ------------------------------------------------------------------

    def _snapshot(self) -> None:
        """Save current fragment state for undo."""
        import copy

        self._history.append((copy.deepcopy(self._fragments), set(self._extra_tracks)))

    def undo(self) -> bool:
        """Restore the previous fragment state. Returns False if nothing to undo."""
        if not self._history:
            return False
        self._fragments, self._extra_tracks = self._history.pop()
        return True

    @property
    def can_undo(self) -> bool:
        """True if there is at least one undo snapshot available."""
        return len(self._history) > 0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def fragments(self) -> List[TrackFragment]:
        """All fragments in the model, including deleted ones."""
        return self._fragments

    @property
    def visible_tracks(self) -> List[int]:
        """All track IDs that currently have at least one (non-deleted) fragment."""
        return sorted(self._occupied_track_ids() | self._extra_tracks)

    @property
    def frame_range(self) -> Tuple[int, int]:
        """Inclusive ``(start, end)`` frame range visible in the editor."""
        return self._view_range

    @property
    def full_frame_range(self) -> Tuple[int, int]:
        """Full frame range loaded into the editor."""
        return self._full_frame_range

    def set_view_range(self, frame_start: int, frame_end: int) -> bool:
        """Update the timeline view window used for coordinate mapping."""
        full_start, full_end = self._full_frame_range
        clamped_start = max(full_start, min(frame_start, full_end))
        clamped_end = max(clamped_start, min(frame_end, full_end))
        new_range = (clamped_start, clamped_end)
        if new_range == self._view_range:
            return False
        self._view_range = new_range
        return True

    def reset_view_range(self) -> bool:
        """Reset the timeline view window to the full loaded frame range."""
        if self._view_range == self._full_frame_range:
            return False
        self._view_range = self._full_frame_range
        return True

    def fragment_by_id(self, frag_id: int) -> Optional[TrackFragment]:
        """Return the fragment with the given ``frag_id``, or ``None`` if not found."""
        for f in self._fragments:
            if f.frag_id == frag_id:
                return f
        return None

    def has_overlap(
        self,
        track_id: int,
        frame_start: int,
        frame_end: int,
        exclude_frag_id: Optional[int] = None,
    ) -> bool:
        """Check whether placing a fragment on *track_id* would overlap."""
        for f in self._fragments:
            if f.deleted or f.track_id != track_id:
                continue
            if exclude_frag_id is not None and f.frag_id == exclude_frag_id:
                continue
            if f.frame_start <= frame_end and f.frame_end >= frame_start:
                return True
        return False

    def add_track_lane(self) -> int:
        """Append a new empty track lane and return its track ID."""
        self._snapshot()
        track_id = self._next_track_id
        self._next_track_id += 1
        self._extra_tracks.add(track_id)
        return track_id

    def _occupied_track_ids(self) -> set[int]:
        return {
            fragment.track_id for fragment in self._fragments if not fragment.deleted
        }

    def _trim_empty_extra_tracks(self) -> None:
        self._extra_tracks.intersection_update(self._occupied_track_ids())

    # ------------------------------------------------------------------
    # Edits
    # ------------------------------------------------------------------

    def split(self, frag_id: int, frame: int) -> bool:
        """Split a fragment at *frame*.

        Creates two fragments: ``[frag.start, frame-1]`` and
        ``[frame, frag.end]``.  Returns False if the split point is at
        a boundary (nothing to split).
        """
        frag = self.fragment_by_id(frag_id)
        if frag is None or frag.deleted:
            return False
        if frame <= frag.frame_start or frame > frag.frame_end:
            return False

        self._snapshot()

        # Shrink existing to [start, frame-1]
        old_end = frag.frame_end
        frag.frame_end = frame - 1

        # Create new fragment [frame, old_end]
        new = TrackFragment(
            frag_id=self._next_frag_id,
            track_id=frag.track_id,
            original_track_id=frag.original_track_id,
            frame_start=frame,
            frame_end=old_end,
        )
        self._next_frag_id += 1
        # Insert right after the split fragment to keep order sensible
        idx = self._fragments.index(frag)
        self._fragments.insert(idx + 1, new)
        return True

    def delete(self, frag_id: int) -> bool:
        """Mark a fragment as deleted."""
        frag = self.fragment_by_id(frag_id)
        if frag is None or frag.deleted:
            return False
        self._snapshot()
        frag.deleted = True
        self._trim_empty_extra_tracks()
        return True

    def reassign(self, frag_id: int, new_track_id: int) -> bool:
        """Move a fragment to a different track lane.

        Returns False if the target lane has an overlapping fragment.
        """
        frag = self.fragment_by_id(frag_id)
        if frag is None or frag.deleted:
            return False
        if frag.track_id == new_track_id:
            return False
        if self.has_overlap(new_track_id, frag.frame_start, frag.frame_end):
            return False
        self._snapshot()
        frag.track_id = new_track_id
        self._trim_empty_extra_tracks()
        return True

    # ------------------------------------------------------------------
    # Compute diff → EditOps
    # ------------------------------------------------------------------

    def compute_ops(self) -> List[EditOp]:
        """Compute the minimal list of edit operations vs. the original state.

        Returns a list of :class:`EditOp` that the correction writer can
        execute to reproduce the user's edits.
        """
        ops: List[EditOp] = []
        for frag in self._fragments:
            if frag.deleted:
                ops.append(
                    EditOp(
                        kind=OpKind.DELETE,
                        track_id=frag.original_track_id,
                        frame_start=frag.frame_start,
                        frame_end=frag.frame_end,
                    )
                )
            elif frag.track_id != frag.original_track_id:
                ops.append(
                    EditOp(
                        kind=OpKind.REASSIGN,
                        track_id=frag.original_track_id,
                        frame_start=frag.frame_start,
                        frame_end=frag.frame_end,
                        new_track_id=frag.track_id,
                    )
                )
        return ops
