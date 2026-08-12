"""Tests for ClassKit AprilTag auto-labeler."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from hydra_suite.classkit.config.presets import apriltag_preset
from hydra_suite.classkit.core.store.db import ClassKitDB


def test_apriltag_preset_labels():
    scheme = apriltag_preset("tag36h11", max_tag_id=9)
    labels = scheme.factors[0].labels
    assert labels[:10] == [f"tag_{i}" for i in range(10)]
    assert labels[-1] == "no_tag"
    assert len(labels) == 11  # tag_0..tag_9 + no_tag


def test_apriltag_preset_total_classes():
    scheme = apriltag_preset("tag36h11", max_tag_id=4)
    assert scheme.total_classes == 6  # tag_0..tag_4 + no_tag


def test_apriltag_preset_single_factor():
    scheme = apriltag_preset("tag36h11", max_tag_id=9)
    assert len(scheme.factors) == 1


def test_apriltag_preset_factor_name_matches_family():
    scheme = apriltag_preset("tag25h9", max_tag_id=5)
    assert scheme.factors[0].name == "tag25h9"


def test_apriltag_preset_name():
    scheme = apriltag_preset("tag36h11", max_tag_id=9)
    assert scheme.name == "apriltag_tag36h11"


def test_apriltag_preset_flat_training_modes_only():
    scheme = apriltag_preset("tag36h11", max_tag_id=9)
    assert scheme.training_modes == ["flat_yolo", "flat_custom"]


def test_apriltag_preset_max_tag_id_zero():
    scheme = apriltag_preset("tag36h11", max_tag_id=0)
    assert scheme.factors[0].labels == ["tag_0", "no_tag"]
    assert scheme.total_classes == 2


def _make_db(tmp_path: Path) -> ClassKitDB:
    """Create a DB with two test images."""
    db = ClassKitDB(tmp_path / "test.db")
    with sqlite3.connect(db.db_path) as conn:
        conn.execute(
            "INSERT INTO images (file_path, file_hash) VALUES (?, ?)",
            ("/img/a.jpg", "aaa"),
        )
        conn.execute(
            "INSERT INTO images (file_path, file_hash) VALUES (?, ?)",
            ("/img/b.jpg", "bbb"),
        )
        conn.commit()
    return db


def test_update_labels_with_confidence_batch_writes_label_and_confidence(tmp_path):
    db = _make_db(tmp_path)
    db.update_labels_with_confidence_batch(
        {
            "/img/a.jpg": ("tag_3", 0.8),
            "/img/b.jpg": ("no_tag", 1.0),
        },
        label_source="auto_apriltag",
        verified=False,
    )
    with sqlite3.connect(db.db_path) as conn:
        rows = conn.execute("""
            SELECT file_path, label, confidence, label_source, verified
            FROM images
            ORDER BY file_path
            """).fetchall()
    assert rows[0] == ("/img/a.jpg", "tag_3", 0.8, "auto_apriltag", 0)
    assert rows[1] == ("/img/b.jpg", "no_tag", 1.0, "auto_apriltag", 0)


def test_update_labels_with_confidence_batch_empty_is_noop(tmp_path):
    db = _make_db(tmp_path)
    db.update_labels_with_confidence_batch({})  # must not raise
    labels = db.get_all_labels()
    assert all(lbl is None for lbl in labels)


def test_clear_all_labels_sets_null(tmp_path):
    db = _make_db(tmp_path)
    db.update_labels_with_confidence_batch(
        {
            "/img/a.jpg": ("tag_3", 0.8),
        }
    )
    db.clear_all_labels()
    with sqlite3.connect(db.db_path) as conn:
        rows = conn.execute(
            "SELECT label, confidence, label_source, verified FROM images"
        ).fetchall()
    assert all(
        row[0] is None and row[1] is None and row[2] is None and row[3] == 0
        for row in rows
    )


def test_update_labels_batch_marks_human_labels_verified(tmp_path):
    db = _make_db(tmp_path)
    db.update_labels_with_confidence_batch(
        {
            "/img/a.jpg": ("tag_3", 0.8),
        },
        label_source="auto_apriltag",
        verified=False,
    )

    db.update_labels_batch({"/img/a.jpg": "tag_7"})

    with sqlite3.connect(db.db_path) as conn:
        row = conn.execute("""
            SELECT label, confidence, label_source, verified, verified_at,
                   auto_label_metadata_json
            FROM images
            WHERE file_path = '/img/a.jpg'
            """).fetchone()

    assert row[0] == "tag_7"
    assert row[1] is None
    assert row[2] == "human"
    assert row[3] == 1
    assert row[4] is not None
    assert row[5] is None


def test_get_labeled_pairs_verified_only_excludes_machine_labels(tmp_path):
    db = _make_db(tmp_path)
    db.update_labels_batch({"/img/a.jpg": "tag_1"})
    db.update_labels_with_confidence_batch(
        {"/img/b.jpg": ("tag_2", 0.9)},
        label_source="auto_apriltag",
        verified=False,
    )

    assert db.get_labeled_pairs() == [("/img/a.jpg", "tag_1"), ("/img/b.jpg", "tag_2")]
    assert db.get_labeled_pairs(verified_only=True) == [("/img/a.jpg", "tag_1")]


def test_mark_labels_verified_preserves_machine_provenance(tmp_path):
    db = _make_db(tmp_path)
    db.update_labels_with_confidence_batch(
        {"/img/a.jpg": ("tag_4", 0.87)},
        label_source="auto_model",
        verified=False,
    )

    updated = db.mark_labels_verified(["/img/a.jpg"])
    status = db.get_label_review_status_by_path()["/img/a.jpg"]

    assert updated == 1
    assert status["label"] == "tag_4"
    assert status["label_source"] == "auto_model"
    assert status["verified"] is True
    assert status["verified_at"] is not None


def test_clear_all_labels_on_empty_db_is_noop(tmp_path):
    db = ClassKitDB(tmp_path / "empty.db")
    db.clear_all_labels()  # must not raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bgr_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a solid gray BGR uint8 image for testing."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _make_observation(tag_id: int, hamming: int = 0):
    """Create a mock TagObservation."""
    obs = MagicMock()
    obs.tag_id = tag_id
    obs.hamming = hamming
    return obs


def _make_config(unsharp_kernel_size=(5, 5), unsharp_sigma=1.0, unsharp_amount=1.5):
    from hydra_suite.core.individual.classification.apriltag import AprilTagConfig

    return AprilTagConfig(
        unsharp_kernel_size=unsharp_kernel_size,
        unsharp_sigma=unsharp_sigma,
        unsharp_amount=unsharp_amount,
    )


# ---------------------------------------------------------------------------
# Preprocessing profile tests
# ---------------------------------------------------------------------------


def test_profile_raw_returns_same_array():
    from hydra_suite.classkit.core.autolabel.apriltag import _profile_raw

    img = _bgr_image()
    out = _profile_raw(img)
    np.testing.assert_array_equal(out, img)


def test_profile_raw_is_3channel_bgr():
    from hydra_suite.classkit.core.autolabel.apriltag import _profile_raw

    out = _profile_raw(_bgr_image())
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8


def test_profile_clahe_is_3channel_bgr():
    from hydra_suite.classkit.core.autolabel.apriltag import _profile_clahe

    out = _profile_clahe(_bgr_image())
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8


def test_profile_clahe_preserves_shape():
    from hydra_suite.classkit.core.autolabel.apriltag import _profile_clahe

    img = _bgr_image(32, 48)
    out = _profile_clahe(img)
    assert out.shape == img.shape


def test_profile_gamma_boost_is_3channel_bgr():
    from hydra_suite.classkit.core.autolabel.apriltag import _profile_gamma_boost

    out = _profile_gamma_boost(_bgr_image())
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8


def test_profile_gamma_boost_brightens_dark_image():
    from hydra_suite.classkit.core.autolabel.apriltag import _profile_gamma_boost

    dark = np.full((8, 8, 3), 50, dtype=np.uint8)
    bright = _profile_gamma_boost(dark)
    assert int(bright.mean()) > int(dark.mean())


def test_profile_gamma_darken_darkens_bright_image():
    from hydra_suite.classkit.core.autolabel.apriltag import _profile_gamma_darken

    bright = np.full((8, 8, 3), 200, dtype=np.uint8)
    dark = _profile_gamma_darken(bright)
    assert int(dark.mean()) < int(bright.mean())


def test_profile_gamma_darken_is_3channel_bgr():
    from hydra_suite.classkit.core.autolabel.apriltag import _profile_gamma_darken

    out = _profile_gamma_darken(_bgr_image())
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8


def test_unsharp_strong_is_3channel_bgr():
    from hydra_suite.classkit.core.autolabel.apriltag import _make_unsharp_strong

    profile = _make_unsharp_strong(_make_config())
    out = profile(_bgr_image())
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8


def test_unsharp_strong_preserves_shape():
    from hydra_suite.classkit.core.autolabel.apriltag import _make_unsharp_strong

    profile = _make_unsharp_strong(_make_config())
    img = _bgr_image(32, 48)
    assert profile(img).shape == img.shape


# ---------------------------------------------------------------------------
# AprilTagAutoLabeler tests (AprilTagDetector is mocked)
# ---------------------------------------------------------------------------


@patch("hydra_suite.classkit.core.autolabel.apriltag.AprilTagDetector")
def test_detector_called_with_internal_config_no_preprocessing(MockDetector):
    """Internal config must have unsharp_amount=0.0 and contrast_factor=1.0."""
    from hydra_suite.classkit.core.autolabel.apriltag import AprilTagAutoLabeler

    config = _make_config()
    mock_inst = MockDetector.return_value
    mock_inst.detect_in_crops.return_value = []

    AprilTagAutoLabeler(config, confidence_threshold=0.6)
    call_args = MockDetector.call_args
    internal_cfg = call_args[0][0]
    assert internal_cfg.unsharp_amount == 0.0
    assert internal_cfg.contrast_factor == 1.0


@patch("hydra_suite.classkit.core.autolabel.apriltag.AprilTagDetector")
def test_detector_wrapped_with_crops_and_offsets(MockDetector):
    """Each profile call must wrap the image as crops=[img], offsets_xy=[(0,0)]."""
    from hydra_suite.classkit.core.autolabel.apriltag import AprilTagAutoLabeler

    mock_inst = MockDetector.return_value
    mock_inst.detect_in_crops.return_value = []

    labeler = AprilTagAutoLabeler(_make_config(), 0.6)
    labeler.label_image(_bgr_image())

    for call in mock_inst.detect_in_crops.call_args_list:
        kwargs = call.kwargs if call.kwargs else {}
        args = call.args
        crops = kwargs.get("crops", args[0] if args else None)
        offsets = kwargs.get("offsets_xy", args[1] if len(args) > 1 else None)
        assert crops is not None and len(crops) == 1
        assert offsets is not None and offsets == [(0, 0)]


@patch("hydra_suite.classkit.core.autolabel.apriltag.AprilTagDetector")
def test_no_tag_when_all_profiles_return_empty(MockDetector):
    from hydra_suite.classkit.core.autolabel.apriltag import AprilTagAutoLabeler

    mock_inst = MockDetector.return_value
    mock_inst.detect_in_crops.return_value = []

    labeler = AprilTagAutoLabeler(_make_config(), 0.6)
    result = labeler.label_image(_bgr_image())

    assert result.label == "no_tag"
    assert result.confidence == 1.0
    assert result.detected_tag_id is None
    assert result.all_no_tag is True


@patch("hydra_suite.classkit.core.autolabel.apriltag.AprilTagDetector")
def test_no_tag_when_all_profiles_ambiguous(MockDetector):
    """Ambiguous (tied hamming) profiles also produce no_tag with confidence 1.0."""
    from hydra_suite.classkit.core.autolabel.apriltag import AprilTagAutoLabeler

    mock_inst = MockDetector.return_value
    mock_inst.detect_in_crops.return_value = [
        _make_observation(tag_id=1, hamming=0),
        _make_observation(tag_id=2, hamming=0),
    ]

    labeler = AprilTagAutoLabeler(_make_config(), 0.6)
    result = labeler.label_image(_bgr_image())

    assert result.label == "no_tag"
    assert result.confidence == 1.0


@patch("hydra_suite.classkit.core.autolabel.apriltag.AprilTagDetector")
def test_confidence_fraction_correct(MockDetector):
    """3 of 5 profiles agree on tag_3 → confidence = 3/5 = 0.6."""
    from hydra_suite.classkit.core.autolabel.apriltag import AprilTagAutoLabeler

    mock_inst = MockDetector.return_value
    call_count = [0]

    def side_effect(crops, offsets_xy, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 3:
            return [_make_observation(tag_id=3, hamming=0)]
        return []

    mock_inst.detect_in_crops.side_effect = side_effect

    labeler = AprilTagAutoLabeler(_make_config(), confidence_threshold=0.5)
    result = labeler.label_image(_bgr_image())

    assert result.label == "tag_3"
    assert abs(result.confidence - 0.6) < 1e-9
    assert result.detected_tag_id == 3


@patch("hydra_suite.classkit.core.autolabel.apriltag.AprilTagDetector")
def test_below_threshold_returns_none_label(MockDetector):
    """1 of 5 profiles agree → confidence=0.2, below threshold=0.6 → label=None."""
    from hydra_suite.classkit.core.autolabel.apriltag import AprilTagAutoLabeler

    mock_inst = MockDetector.return_value
    call_count = [0]

    def side_effect(crops, offsets_xy, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return [_make_observation(tag_id=5, hamming=0)]
        return []

    mock_inst.detect_in_crops.side_effect = side_effect

    labeler = AprilTagAutoLabeler(_make_config(), confidence_threshold=0.6)
    result = labeler.label_image(_bgr_image())

    assert result.label is None
    assert abs(result.confidence - 0.2) < 1e-9


@patch("hydra_suite.classkit.core.autolabel.apriltag.AprilTagDetector")
def test_multi_tag_tie_discards_profile(MockDetector):
    """Two observations with same hamming → AMBIGUOUS → profile counts as no detection."""
    from hydra_suite.classkit.core.autolabel.apriltag import AprilTagAutoLabeler

    mock_inst = MockDetector.return_value
    call_count = [0]

    def side_effect(crops, offsets_xy, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return [
                _make_observation(tag_id=1, hamming=0),
                _make_observation(tag_id=2, hamming=0),
            ]
        return [_make_observation(tag_id=7, hamming=0)]

    mock_inst.detect_in_crops.side_effect = side_effect

    labeler = AprilTagAutoLabeler(_make_config(), confidence_threshold=0.5)
    result = labeler.label_image(_bgr_image())

    assert result.label == "tag_7"
    assert abs(result.confidence - 0.8) < 1e-9


@patch("hydra_suite.classkit.core.autolabel.apriltag.AprilTagDetector")
def test_multi_tag_clear_winner_used(MockDetector):
    """Profile with two observations, one has lower hamming → that tag_id is used."""
    from hydra_suite.classkit.core.autolabel.apriltag import AprilTagAutoLabeler

    mock_inst = MockDetector.return_value
    mock_inst.detect_in_crops.return_value = [
        _make_observation(tag_id=1, hamming=1),  # higher hamming → loser
        _make_observation(tag_id=2, hamming=0),  # lower hamming → winner
    ]

    labeler = AprilTagAutoLabeler(_make_config(), confidence_threshold=0.1)
    result = labeler.label_image(_bgr_image())

    assert result.detected_tag_id == 2
    assert result.label == "tag_2"


@patch("hydra_suite.classkit.core.autolabel.apriltag.AprilTagDetector")
def test_majority_vote_wins(MockDetector):
    """3 profiles return tag_1, 2 profiles return tag_2 → tag_1 wins."""
    from hydra_suite.classkit.core.autolabel.apriltag import AprilTagAutoLabeler

    mock_inst = MockDetector.return_value
    call_count = [0]

    def side_effect(crops, offsets_xy, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 3:
            return [_make_observation(tag_id=1, hamming=0)]
        return [_make_observation(tag_id=2, hamming=0)]

    mock_inst.detect_in_crops.side_effect = side_effect

    labeler = AprilTagAutoLabeler(_make_config(), confidence_threshold=0.5)
    result = labeler.label_image(_bgr_image())

    assert result.detected_tag_id == 1
    assert result.label == "tag_1"


@patch("hydra_suite.classkit.core.autolabel.apriltag.AprilTagDetector")
def test_autolabel_images_empty_list(MockDetector):
    from hydra_suite.classkit.core.autolabel.apriltag import autolabel_images

    results = autolabel_images([], _make_config(), threshold=0.6)
    assert results == []
    MockDetector.assert_not_called()


@patch("hydra_suite.classkit.core.autolabel.apriltag.AprilTagDetector")
def test_autolabel_images_returns_one_result_per_path(MockDetector, tmp_path):
    import cv2 as _cv2

    from hydra_suite.classkit.core.autolabel.apriltag import autolabel_images

    mock_inst = MockDetector.return_value
    mock_inst.detect_in_crops.return_value = []

    for name in ("a.jpg", "b.jpg"):
        _cv2.imwrite(str(tmp_path / name), _bgr_image())

    results = autolabel_images(
        [tmp_path / "a.jpg", tmp_path / "b.jpg"], _make_config(), threshold=0.6
    )
    assert len(results) == 2
