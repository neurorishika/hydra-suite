"""Tests for cache identity hash functions in properties_cache."""

from __future__ import annotations

from tests.helpers.module_loader import load_src_module

mod = load_src_module(
    "hydra_suite/core/identity/properties/cache.py",
    "properties_cache_under_test",
)


# ---- compute_apriltag_cache_id ----


def test_apriltag_cache_id_deterministic() -> None:
    params = {
        "APRILTAG_FAMILY": "tag36h11",
        "APRILTAG_MAX_HAMMING": 1,
        "APRILTAG_DECIMATE": 1.0,
        "APRILTAG_BLUR": 0.8,
        "INDIVIDUAL_CROP_PADDING": 0.1,
    }
    id1 = mod.compute_apriltag_cache_id(params, "model_a")
    id2 = mod.compute_apriltag_cache_id(params, "model_a")
    assert id1 == id2


def test_apriltag_cache_id_changes_with_family() -> None:
    base = {"APRILTAG_FAMILY": "tag36h11", "APRILTAG_MAX_HAMMING": 1}
    alt = {"APRILTAG_FAMILY": "tag25h9", "APRILTAG_MAX_HAMMING": 1}
    assert mod.compute_apriltag_cache_id(base, "m") != mod.compute_apriltag_cache_id(
        alt, "m"
    )


def test_apriltag_cache_id_changes_with_hamming() -> None:
    base = {"APRILTAG_MAX_HAMMING": 1}
    alt = {"APRILTAG_MAX_HAMMING": 2}
    assert mod.compute_apriltag_cache_id(base, "m") != mod.compute_apriltag_cache_id(
        alt, "m"
    )


def test_apriltag_cache_id_changes_with_decimate() -> None:
    base = {"APRILTAG_DECIMATE": 1.0}
    alt = {"APRILTAG_DECIMATE": 2.0}
    assert mod.compute_apriltag_cache_id(base, "m") != mod.compute_apriltag_cache_id(
        alt, "m"
    )


def test_apriltag_cache_id_changes_with_inference_model() -> None:
    params = {"APRILTAG_FAMILY": "tag36h11"}
    assert mod.compute_apriltag_cache_id(
        params, "yolo_abc"
    ) != mod.compute_apriltag_cache_id(params, "yolo_xyz")


def test_apriltag_cache_id_changes_with_padding() -> None:
    base = {"INDIVIDUAL_CROP_PADDING": 0.1}
    alt = {"INDIVIDUAL_CROP_PADDING": 0.2}
    assert mod.compute_apriltag_cache_id(base, "m") != mod.compute_apriltag_cache_id(
        alt, "m"
    )


def test_apriltag_cache_id_changes_with_preprocessing() -> None:
    base = {"APRILTAG_CONTRAST_FACTOR": 1.5}
    alt = {"APRILTAG_CONTRAST_FACTOR": 2.0}
    assert mod.compute_apriltag_cache_id(base, "m") != mod.compute_apriltag_cache_id(
        alt, "m"
    )


# ---- compute_classify_cache_id ----


def test_classify_cache_id_deterministic() -> None:
    id1 = mod.compute_classify_cache_id("/tmp/m.pth", "cpu", "model_a")
    id2 = mod.compute_classify_cache_id("/tmp/m.pth", "cpu", "model_a")
    assert id1 == id2


def test_classify_cache_id_changes_with_model() -> None:
    id1 = mod.compute_classify_cache_id("/tmp/m1.pth", "cpu", "det_a")
    id2 = mod.compute_classify_cache_id("/tmp/m2.pth", "cpu", "det_a")
    assert id1 != id2


def test_classify_cache_id_changes_with_runtime() -> None:
    id1 = mod.compute_classify_cache_id("/tmp/m.pth", "cpu", "det_a")
    id2 = mod.compute_classify_cache_id("/tmp/m.pth", "mps", "det_a")
    assert id1 != id2


def test_classify_cache_id_changes_with_inference_model() -> None:
    id1 = mod.compute_classify_cache_id("/tmp/m.pth", "cpu", "det_a")
    id2 = mod.compute_classify_cache_id("/tmp/m.pth", "cpu", "det_b")
    assert id1 != id2


def test_classify_cache_id_changes_with_calibration_signature() -> None:
    id1 = mod.compute_classify_cache_id(
        "/tmp/m.pth",
        "cpu",
        "det_a",
        calibration_signature="temp:1.25",
    )
    id2 = mod.compute_classify_cache_id(
        "/tmp/m.pth",
        "cpu",
        "det_a",
        calibration_signature="temp:1.75",
    )
    assert id1 != id2
