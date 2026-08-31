"""One provider per overlay data source.

The canvas used to conflate rendering shapes with knowing what ground
truth, predictions and staged escalations ARE -- and the three genuinely
differ (class-id space, whether confidence exists, lifecycle, geometry
level), so those differences leaked into show_image. Each quirk now lives
in exactly one small class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from PySide6.QtCore import Qt

from hydra_suite.utils.geometry_levels import GeometryLevel

from ..colors import ESCALATION_COLOUR
from ..utils import (
    find_label_for_image,
    find_staged_label_for_image,
    parse_obb_label,
    source_class_id_map,
    staged_class_names,
)
from .layer import ColourPolicy, Emphasis, LabelMode, LayerStyle, OverlayLayer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameContext:
    """Everything every provider needs about the frame on screen."""

    project: Any
    source_path: str
    image_path: str
    # (h, w) taken from the LOADED PIXMAP, never decoded again. Re-decoding
    # the file per provider cost ~100 ms per keypress on 4512^2 frames.
    size: tuple[int, int]
    predictions: list[dict] = field(default_factory=list)

    def source(self):
        if self.project is None:
            return None
        return next(
            (s for s in self.project.sources if str(s.path) == str(self.source_path)),
            None,
        )


class OverlayProvider(Protocol):
    key: str

    def build(self, ctx: FrameContext) -> Optional[OverlayLayer]: ...  # noqa: E704


def resolve_pending_level(pending):
    """Geometry level a staged escalation's labels are in.

    ``StagedReview.target_level`` is load-bearing: SAM2 converts
    existing boxes IN PLACE and can stage OBB, while SAM3 stages polygons.
    Drawing an OBB quad as polygon-native gave it the polygon style AND a
    derived OBB of the same quad -- a duplicate outline in the wrong style.

    Like ``OBBSource.level`` this is an unvalidated string from project
    JSON, so it degrades rather than raising: an unparseable value falls
    back to POLYGON, which is what both producers stage by default.
    """
    raw = str(getattr(pending, "target_level", "") or "")
    try:
        return GeometryLevel.from_str(raw)
    except ValueError:
        logger.warning(
            "Unknown target_level %r on a staged escalation; rendering as polygon",
            raw,
        )
        return GeometryLevel.POLYGON


def resolve_source_render_state(project, source_path):
    """Return (native_level, reviewed) for the OBBSource at *source_path* in
    *project*. Falls back to (GeometryLevel.OBB, True) if project is None,
    no source matches, or the matched source's level string doesn't parse --
    OBBSource.level is an unvalidated string loaded from project JSON, so a
    hand-edited or future-version file must degrade gracefully here rather
    than crashing show_image on every image selection."""
    if project is None:
        return GeometryLevel.OBB, True

    src_obj = next((s for s in project.sources if s.path == source_path), None)
    if src_obj is None:
        return GeometryLevel.OBB, True

    try:
        native_level = GeometryLevel.from_str(src_obj.level)
    except ValueError:
        logger.warning(
            "Unknown geometry level %r for source %r; rendering as OBB",
            src_obj.level,
            source_path,
        )
        native_level = GeometryLevel.OBB

    return native_level, src_obj.reviewed


class GroundTruthProvider:
    key = "gt"

    def build(self, ctx: FrameContext) -> Optional[OverlayLayer]:
        label_path = find_label_for_image(Path(ctx.image_path), ctx.source_path)
        if label_path is None:
            return None
        h, w = ctx.size
        class_names = ctx.project.class_names if ctx.project is not None else ["object"]
        class_id_map = None
        if ctx.project is not None:
            try:
                class_id_map = source_class_id_map(ctx.source_path, class_names)
            except Exception:
                class_id_map = {}
                logger.warning(
                    "Skipping incompatible source labels for preview: %s",
                    ctx.source_path,
                    exc_info=True,
                )
        dets = parse_obb_label(label_path, w, h, class_id_map=class_id_map)
        native_level, reviewed = resolve_source_render_state(
            ctx.project, ctx.source_path
        )
        return OverlayLayer(
            key=self.key,
            detections=dets,
            native_level=native_level,
            class_names=class_names,
            colour_policy=ColourPolicy.PER_CLASS,
            label_mode=LabelMode.NAME_AND_CLASS_ID,
            emphasis=None if reviewed else Emphasis.UNREVIEWED,
            z=0,
        )


class PredictionProvider:
    key = "pred"

    def build(self, ctx: FrameContext) -> Optional[OverlayLayer]:
        if not ctx.predictions:
            return None
        class_names = ctx.project.class_names if ctx.project is not None else ["object"]
        return OverlayLayer(
            key=self.key,
            detections=list(ctx.predictions),
            native_level=GeometryLevel.AABB,
            class_names=class_names,
            colour_policy=ColourPolicy.PER_CLASS,
            derive_levels=False,
            style=LayerStyle(Qt.PenStyle.DashLine, Qt.BrushStyle.NoBrush, 0),
            label_mode=LabelMode.NAME_AND_CONFIDENCE,
            z=20,
        )


class StagedReviewProvider:
    key = "staged"

    def build(self, ctx: FrameContext) -> Optional[OverlayLayer]:
        source = ctx.source()
        review = getattr(source, "staged_review", None) if source else None
        if review is None or not str(getattr(review, "staged_path", "")).strip():
            return None
        label_path = find_staged_label_for_image(
            Path(ctx.image_path), ctx.source_path, review.staged_path
        )
        if label_path is None:
            return None

        # A decided frame's proposal is resolved: an accepted one now lives on
        # the ground-truth layer, a rejected one is not wanted. Keeping it in
        # magenta would make a reviewed frame look identical to an unreviewed
        # one, which is the single thing frame-granular review has to show.
        from ...jobs.staged_review import read_decisions, review_key_for_image

        # The key comes from the IMAGE, not from `label_path`.
        # find_staged_label_for_image has stem and recursive fallbacks, so
        # label_path.relative_to(staged_labels) is not guaranteed to be the
        # same string staged_frames/accept_frame use as their key. Deriving
        # it from the image is the only way the three agree.
        rel = review_key_for_image(ctx.source_path, ctx.image_path)
        if rel is not None and rel in read_decisions(review.staged_path):
            return None

        h, w = ctx.size
        # No class_id_map: staged ids index the STAGING dir's classes.txt,
        # not the project's class list, so remapping them would mislabel.
        dets = parse_obb_label(label_path, w, h)
        if not dets:
            return None
        return OverlayLayer(
            key=self.key,
            detections=dets,
            native_level=resolve_pending_level(review),
            class_names=staged_class_names(review.staged_path),
            colour_policy=ColourPolicy.FIXED,
            fixed_colour=ESCALATION_COLOUR,
            class_filtered=False,
            label_mode=LabelMode.NAME_AND_CONFIDENCE,
            z=10,
        )


# Draw order == today's show_image order (GT, escalation, predictions), and
# the z values above encode the same stacking.
PROVIDERS: tuple = (
    GroundTruthProvider(),
    StagedReviewProvider(),
    PredictionProvider(),
)
