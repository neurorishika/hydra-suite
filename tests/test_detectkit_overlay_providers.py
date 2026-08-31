"""Behavioural tests for the overlay providers.

These replace the inspect.getsource assertions that used to stand in for
them: a provider is a plain object that can be called without a
MainWindow, so the tests check what it BUILDS rather than what its
caller's source text contains.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from hydra_suite.detectkit.gui.overlays import (  # noqa: E402
    ColourPolicy,
    Emphasis,
    FrameContext,
    GroundTruthProvider,
    LabelMode,
    PredictionProvider,
    StagedReviewProvider,
    resolve_pending_level,
    resolve_source_render_state,
)
from hydra_suite.utils.geometry_levels import GeometryLevel  # noqa: E402


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def source_tree(tmp_path):
    src = tmp_path / "src_a"
    (src / "images").mkdir(parents=True)
    img = src / "images" / "f0001.png"
    img.write_bytes(b"")
    _write(src / "labels" / "f0001.txt", ["0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3"])
    # REQUIRED: source_class_id_map -> read_classes_txt raises RuntimeError
    # without it, the provider's except-branch zeroes the class map, and
    # parse_obb_label then drops every line. A missing classes.txt makes
    # these tests fail in a way that looks like a provider bug.
    (src / "classes.txt").write_text("ant\nworker\n")
    return SimpleNamespace(root=src, image=img)


def _project(source_tree, **kw):
    source = SimpleNamespace(
        path=str(source_tree.root),
        name="src_a",
        level="polygon",
        reviewed=True,
        staged_review=None,
    )
    for k, v in kw.items():
        setattr(source, k, v)
    return SimpleNamespace(class_names=["ant", "worker"], sources=[source])


def _ctx(project, source_tree, **kw):
    base = dict(
        project=project,
        source_path=str(source_tree.root),
        image_path=str(source_tree.image),
        size=(100, 100),
        predictions=[],
    )
    base.update(kw)
    return FrameContext(**base)


def test_ground_truth_provider_builds_a_per_class_multi_level_layer(source_tree):
    layer = GroundTruthProvider().build(_ctx(_project(source_tree), source_tree))
    assert layer.key == "gt"
    assert layer.colour_policy is ColourPolicy.PER_CLASS
    assert layer.label_mode is LabelMode.NAME_AND_CLASS_ID
    assert layer.derive_levels is True
    assert layer.class_filtered is True
    assert layer.native_level is GeometryLevel.POLYGON
    assert layer.emphasis is None
    assert layer.z == 0
    assert len(layer.detections) == 1


def test_ground_truth_provider_flags_an_unreviewed_source(source_tree):
    layer = GroundTruthProvider().build(
        _ctx(_project(source_tree, reviewed=False), source_tree)
    )
    assert layer.emphasis is Emphasis.UNREVIEWED


def test_ground_truth_provider_returns_none_when_the_frame_has_no_label(source_tree):
    (source_tree.root / "labels" / "f0001.txt").unlink()
    assert GroundTruthProvider().build(_ctx(_project(source_tree), source_tree)) is None


def test_prediction_provider_labels_with_confidence_and_never_derives(source_tree):
    preds = [{"class_id": 0, "polygon_px": [(1, 1), (5, 1), (5, 5)], "confidence": 0.5}]
    layer = PredictionProvider().build(
        _ctx(_project(source_tree), source_tree, predictions=preds)
    )
    assert layer.key == "pred"
    assert layer.label_mode is LabelMode.NAME_AND_CONFIDENCE
    assert layer.derive_levels is False
    assert layer.style is not None
    assert layer.z == 20


def test_prediction_provider_returns_none_with_no_predictions(source_tree):
    assert PredictionProvider().build(_ctx(_project(source_tree), source_tree)) is None


def _staged(tmp_path, target_level):
    staged = tmp_path / "staged"
    # labels/f0001.txt, NOT labels/images/f0001.txt: the staged tree mirrors
    # the source's images/ tree, and that mirroring IS the review key. The
    # old nesting only resolved via find_staged_label_for_image's recursive
    # fallback, which no longer agrees with the decisions key.
    _write(
        staged / "labels" / "f0001.txt",
        ["0 0.2 0.2 0.4 0.2 0.4 0.4 0.2 0.4"],
    )
    (staged / "classes.txt").write_text("prompt_a\n")
    return SimpleNamespace(staged_path=str(staged), target_level=target_level)


def test_staged_provider_is_fixed_colour_and_unfiltered(source_tree, tmp_path):
    project = _project(source_tree, staged_review=_staged(tmp_path, "obb"))
    layer = StagedReviewProvider().build(_ctx(project, source_tree))
    assert layer.key == "escalation"
    assert layer.colour_policy is ColourPolicy.FIXED
    assert layer.fixed_colour is not None
    assert layer.class_filtered is False
    assert layer.native_level is GeometryLevel.OBB
    assert layer.label_mode is LabelMode.NAME_AND_CONFIDENCE
    assert layer.z == 10


def test_staged_provider_honours_the_escalations_own_target_level(
    source_tree, tmp_path
):
    """A SAM2 run can stage OBB. Hardcoding POLYGON here once gave a staged
    OBB polygon styling plus a duplicate derived OBB outline."""
    project = _project(source_tree, staged_review=_staged(tmp_path, "aabb"))
    layer = StagedReviewProvider().build(_ctx(project, source_tree))
    assert layer.native_level is GeometryLevel.AABB


def test_staged_provider_matches_source_across_str_and_path_types(
    source_tree, tmp_path
):
    """FrameContext.source() must coerce both sides with str() before
    comparing, matching main_window.py's _refresh_escalation_overlay
    lookup (str(s.path) == str(source_path)). If a Path ever reaches
    either side, a bare == would silently drop the escalation overlay."""
    project = _project(source_tree, staged_review=_staged(tmp_path, "obb"))
    project.sources[0].path = Path(project.sources[0].path)
    ctx = _ctx(project, source_tree)
    assert isinstance(ctx.source_path, str)
    assert isinstance(project.sources[0].path, Path)
    layer = StagedReviewProvider().build(ctx)
    assert layer is not None
    assert layer.key == "escalation"


def test_staged_provider_returns_none_without_a_pending_escalation(source_tree):
    assert (
        StagedReviewProvider().build(_ctx(_project(source_tree), source_tree)) is None
    )


def test_the_escalation_layer_stacks_below_predictions(source_tree, tmp_path):
    """Pins a DELIBERATE behaviour change, not pre-existing characterization.

    Pre-refactor, stacking was pure scene-insertion order (no `setZValue`
    anywhere in the old canvas): a plain `show_image` put staged escalation
    masks BELOW predictions, but the post-dialog
    `escalation_actions._refresh_escalation_overlay()` re-inserted the
    escalation layer last, putting it ABOVE predictions on that one path --
    i.e. the old behaviour was call-order-dependent and self-contradictory.
    The registry's explicit z-order (gt=0, escalation=10, pred=20) makes
    stacking consistent on every path: predictions always render above
    staged masks. See the "Known deviations from pixel-identical" section
    of docs/superpowers/specs/2026-08-31-detectkit-overlay-layer-registry-design.md
    for the full rationale."""
    project = _project(source_tree, staged_review=_staged(tmp_path, "obb"))
    ctx = _ctx(
        project,
        source_tree,
        predictions=[{"class_id": 0, "polygon_px": [(1, 1), (5, 1), (5, 5)]}],
    )
    gt = GroundTruthProvider().build(ctx)
    esc = StagedReviewProvider().build(ctx)
    pred = PredictionProvider().build(ctx)
    assert gt.z < esc.z < pred.z


def test_staged_layer_disappears_once_the_frame_is_decided(source_tree, tmp_path):
    from hydra_suite.detectkit.jobs.staged_review import (
        ACCEPTED_ADD_NEW,
        review_key_for_image,
        write_decisions,
    )

    staged = _staged(tmp_path, "obb")
    project = _project(source_tree, staged_review=staged)
    ctx = _ctx(project, source_tree)
    assert StagedReviewProvider().build(ctx) is not None

    rel = review_key_for_image(source_tree.root, ctx.image_path)
    assert rel == "f0001.txt"
    write_decisions(staged.staged_path, {rel: ACCEPTED_ADD_NEW})

    assert StagedReviewProvider().build(ctx) is None


def test_a_decision_on_another_frame_does_not_hide_this_one(source_tree, tmp_path):
    from hydra_suite.detectkit.jobs.staged_review import REJECTED, write_decisions

    staged = _staged(tmp_path, "obb")
    project = _project(source_tree, staged_review=staged)
    write_decisions(staged.staged_path, {"some/other/frame.txt": REJECTED})

    assert StagedReviewProvider().build(_ctx(project, source_tree)) is not None


def test_resolve_native_level_and_reviewed_reads_the_matching_source():
    """Behavioral check (not just source-text): the level/reviewed resolved
    for a given source_path must actually come from the OBBSource whose
    .path matches, not some other source or a wrong default."""
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
    from hydra_suite.training.geometry_levels import GeometryLevel

    proj = DetectKitProject(class_names=["ant"])
    proj.sources = [
        OBBSource(path="/a", name="a", level="obb", reviewed=True),
        OBBSource(path="/b", name="b", level="polygon", reviewed=False),
    ]

    native_level, reviewed = resolve_source_render_state(proj, "/b")
    assert native_level == GeometryLevel.POLYGON
    assert reviewed is False


def test_resolve_native_level_and_reviewed_defaults_when_source_missing():
    from hydra_suite.detectkit.gui.models import DetectKitProject
    from hydra_suite.training.geometry_levels import GeometryLevel

    proj = DetectKitProject(class_names=["ant"])
    native_level, reviewed = resolve_source_render_state(proj, "/nonexistent")
    assert native_level == GeometryLevel.OBB
    assert reviewed is True


def test_resolve_native_level_and_reviewed_falls_back_on_unknown_level_string():
    """A hand-edited/future-version project JSON could carry a level string
    GeometryLevel.from_str doesn't recognize -- this must degrade to OBB
    with a warning, not crash show_image on every image selection."""
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
    from hydra_suite.training.geometry_levels import GeometryLevel

    proj = DetectKitProject(class_names=["ant"])
    proj.sources = [
        OBBSource(path="/c", name="c", level="not_a_real_level", reviewed=True)
    ]

    native_level, reviewed = resolve_source_render_state(proj, "/c")
    assert native_level == GeometryLevel.OBB  # fallback, not a raised ValueError
    assert reviewed is True


def test_pending_level_parses_the_record_and_degrades_on_junk():
    """target_level is an unvalidated string from project JSON, exactly like
    OBBSource.level -- a hand-edited or future-version file must not crash
    the overlay on every image selection."""
    from hydra_suite.detectkit.gui.models import StagedReview
    from hydra_suite.training.geometry_levels import GeometryLevel

    assert resolve_pending_level(StagedReview(target_level="obb")) == GeometryLevel.OBB
    assert (
        resolve_pending_level(StagedReview(target_level="polygon"))
        == GeometryLevel.POLYGON
    )
    # Junk degrades to POLYGON: a staged mask is a polygon unless the record
    # says otherwise, and both escalation producers stage polygons by default.
    assert (
        resolve_pending_level(StagedReview(target_level="not_a_level"))
        == GeometryLevel.POLYGON
    )
