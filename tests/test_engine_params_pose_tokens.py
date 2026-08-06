"""Task 5 of the shared engine-param-builder program: lock the two pose

behaviors that diverge between the GUI's ``get_parameters_dict()`` and the
shared Qt-free ``build_engine_params`` in the *general* case (not exercised
by the 7 equivalence-gate fixtures):

1. Pose keypoint tokens (``POSE_IGNORE_KEYPOINTS`` /
   ``POSE_DIRECTION_ANTERIOR_KEYPOINTS`` / ``POSE_DIRECTION_POSTERIOR_KEYPOINTS``)
   must be plain strings, matching the bridge's
   ``MainWindow._selected_pose_group_keypoints`` (``gui/main_window.py:1159-1167``)
   -- never int-coerced, even for numeric-looking keypoint names.
2. ``POSE_SLEAP_BATCH`` must equal ``POSE_BATCH_SIZE`` / ``POSE_YOLO_BATCH``
   (all three come from the single ``spin_pose_batch.value()`` widget --
   bridge ``config.py:2472-2475``), even when the config carries a distinct
   ``pose_sleap_batch`` value.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.trackerkit.engine_params import RuntimeContext, build_engine_params


def _runtime() -> RuntimeContext:
    return RuntimeContext(fps=30.0, total_frames=100, frame_width=640, frame_height=480)


def test_pose_keypoint_tokens_stay_plain_strings():
    config = {
        "pose_ignore_keypoints": ["3", "head"],
        "pose_direction_anterior_keypoints": ["1", "nose"],
        "pose_direction_posterior_keypoints": ["2", "tail"],
    }

    params = build_engine_params(config, runtime=_runtime())

    assert params["POSE_IGNORE_KEYPOINTS"] == ["3", "head"]
    assert params["POSE_DIRECTION_ANTERIOR_KEYPOINTS"] == ["1", "nose"]
    assert params["POSE_DIRECTION_POSTERIOR_KEYPOINTS"] == ["2", "tail"]

    for key in (
        "POSE_IGNORE_KEYPOINTS",
        "POSE_DIRECTION_ANTERIOR_KEYPOINTS",
        "POSE_DIRECTION_POSTERIOR_KEYPOINTS",
    ):
        for token in params[key]:
            assert isinstance(
                token, str
            ), f"{key} token {token!r} is not a plain string"


def test_pose_sleap_batch_matches_pose_batch_size():
    config = {
        "pose_batch_size": 8,
        "pose_sleap_batch": 99,
    }

    params = build_engine_params(config, runtime=_runtime())

    assert params["POSE_BATCH_SIZE"] == 8
    assert params["POSE_YOLO_BATCH"] == 8
    assert params["POSE_SLEAP_BATCH"] == 8
