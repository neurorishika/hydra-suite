"""Tests for per-frame tag feature helpers (tag_features.py)."""

from __future__ import annotations

import numpy as np

from tests.helpers.module_loader import load_src_module

mod = load_src_module(
    "hydra_suite/core/tracking/tag_features.py",
    "tag_features_under_test",
)
NO_TAG = mod.NO_TAG
build_tag_detection_map = mod.build_tag_detection_map
build_tag_detection_hamming_map = mod.build_tag_detection_hamming_map
build_detection_tag_id_list = mod.build_detection_tag_id_list
get_detection_tag_csv_values = mod.get_detection_tag_csv_values


# --- Shared fake cache ---


class FakeTagCache:
    """Minimal stand-in for TagObservationCache in read mode."""

    def __init__(self, data):
        self._data = data

    def get_frame(self, frame_idx):
        empty = {
            "tag_ids": np.array([], dtype=np.int32),
            "det_indices": np.array([], dtype=np.int32),
            "hammings": np.array([], dtype=np.int32),
        }
        return self._data.get(frame_idx, empty)


# --- build_tag_detection_map ---


def test_build_tag_detection_map_basic():
    cache = FakeTagCache(
        {
            5: {
                "tag_ids": np.array([10, 20], dtype=np.int32),
                "det_indices": np.array([0, 3], dtype=np.int32),
                "hammings": np.array([0, 1], dtype=np.int32),
            }
        }
    )
    result = build_tag_detection_map(cache, 5)
    assert result == {0: 10, 3: 20}


def test_build_tag_detection_map_none_cache():
    result = build_tag_detection_map(None, 0)
    assert result == {}


# --- build_tag_detection_hamming_map ---


def test_build_tag_detection_hamming_map_basic():
    cache = FakeTagCache(
        {
            7: {
                "tag_ids": np.array([10, 20], dtype=np.int32),
                "det_indices": np.array([0, 3], dtype=np.int32),
                "hammings": np.array([0, 2], dtype=np.int32),
            }
        }
    )
    result = build_tag_detection_hamming_map(cache, 7)
    assert result == {0: 0, 3: 2}


def test_build_tag_detection_hamming_map_none_cache():
    result = build_tag_detection_hamming_map(None, 0)
    assert result == {}


def test_build_tag_detection_hamming_map_missing_frame():
    cache = FakeTagCache({})
    result = build_tag_detection_hamming_map(cache, 99)
    assert result == {}


# --- build_detection_tag_id_list ---


def test_build_detection_tag_id_list_basic():
    tag_map = {0: 10, 3: 20}
    result = build_detection_tag_id_list(tag_map, 5)
    assert result == [10, NO_TAG, NO_TAG, 20, NO_TAG]


def test_build_detection_tag_id_list_empty():
    result = build_detection_tag_id_list({}, 3)
    assert result == [NO_TAG, NO_TAG, NO_TAG]


# --- get_detection_tag_csv_values ---


def test_get_detection_tag_csv_values_hit():
    tag_det_map = {2: 10}
    tag_hamming_map = {2: 0}
    tag_label_map = {10: "ant1"}
    tag_id, label, conf, hamming = get_detection_tag_csv_values(
        2, tag_det_map, tag_hamming_map, tag_label_map
    )
    assert tag_id == 10.0
    assert label == "ant1"
    assert conf == 1.0
    assert hamming == 0.0


def test_get_detection_tag_csv_values_miss():
    tag_id, label, conf, hamming = get_detection_tag_csv_values(5, {}, {}, {})
    import math

    assert math.isnan(tag_id)
    assert math.isnan(label)
    assert math.isnan(conf)
    assert math.isnan(hamming)


def test_get_detection_tag_csv_values_hamming_nonzero():
    tag_det_map = {1: 20}
    tag_hamming_map = {1: 3}
    tag_label_map = {20: "ant2"}
    tag_id, label, conf, hamming = get_detection_tag_csv_values(
        1, tag_det_map, tag_hamming_map, tag_label_map
    )
    assert tag_id == 20.0
    assert label == "ant2"
    assert conf == 1.0 / (1.0 + 3)
    assert hamming == 3.0


def test_get_detection_tag_csv_values_unknown_label():
    """Tag ID present but not in label map → label is NaN."""
    tag_det_map = {0: 99}
    tag_hamming_map = {0: 0}
    tag_label_map = {10: "ant1"}
    _, label, _, _ = get_detection_tag_csv_values(
        0, tag_det_map, tag_hamming_map, tag_label_map
    )
    import math

    assert math.isnan(label)
