"""User-mode clean-CSV golden tests.

Goldens under ``tests/goldens/user_mode/`` are captured from REAL runs of the
User-mode pipeline (``debug_mode=False``) against the equivalence-harness
fixture clips (see ``tools/equivalence/fixtures/``), via
``tools/equivalence/runner.py`` on the worktree's own ``src`` (PYTHONPATH),
never hand-written.

Note on schema: all three source fixture configs have
``enable_identity_analysis=True`` (the identity-postprocessing / rich-export
path always resolves an ``IdentityFinalLabel`` column when that pipeline is
enabled, regardless of ``identity_method``), so even the "pure tracking"
``fly_obb`` fixture carries the ``identity``/``identity_confidence``/
``identity_source`` columns -- populated with placeholder values
(``identity="unknown"``, ``identity_confidence=0.0``, empty source) since its
``identity_method`` is ``none_disabled``. ``ant_cnn_identity`` additionally
resolves real per-track identities (``identity_method=cnn_classifier``) AND
carries pose keypoint triples, since its fixture config also supplies a
skeleton/pose backend. ``ant_pose_headtail`` carries real pose keypoint
triples via SLEAP head-tail pose.
"""

import pandas as pd
import pytest

CORE_COLUMNS = [
    "id",
    "frame",
    "time_s",
    "x",
    "y",
    "heading_deg",
    "state",
    "detection_confidence",
]

IDENTITY_COLUMNS = ["identity", "identity_confidence", "identity_source"]

POSE_TRIPLE_COLUMNS = [
    "left_antenna_tip_x",
    "left_antenna_tip_y",
    "left_antenna_tip_conf",
    "right_antenna_tip_x",
    "right_antenna_tip_y",
    "right_antenna_tip_conf",
    "left_antenna_elbow_x",
    "left_antenna_elbow_y",
    "left_antenna_elbow_conf",
    "right_antenna_elbow_x",
    "right_antenna_elbow_y",
    "right_antenna_elbow_conf",
    "clypeus_x",
    "clypeus_y",
    "clypeus_conf",
    "neck_x",
    "neck_y",
    "neck_conf",
    "petiole_post_petiole_x",
    "petiole_post_petiole_y",
    "petiole_post_petiole_conf",
    "tip_of_gaster_x",
    "tip_of_gaster_y",
    "tip_of_gaster_conf",
]

CASES = {
    "fly_obb": CORE_COLUMNS + IDENTITY_COLUMNS,
    "ant_cnn_identity": CORE_COLUMNS + IDENTITY_COLUMNS + POSE_TRIPLE_COLUMNS,
    "ant_pose_headtail": CORE_COLUMNS + IDENTITY_COLUMNS + POSE_TRIPLE_COLUMNS,
}


@pytest.mark.parametrize("clip,cols", CASES.items())
def test_user_mode_columns(clip, cols):
    golden = pd.read_csv(f"tests/goldens/user_mode/{clip}_tracks.csv")
    assert list(golden.columns) == cols


def test_user_mode_goldens_nonempty():
    for clip in ("fly_obb", "ant_cnn_identity", "ant_pose_headtail"):
        golden = pd.read_csv(f"tests/goldens/user_mode/{clip}_tracks.csv")
        assert len(golden) > 0


def test_fly_obb_core_columns_present():
    golden = pd.read_csv("tests/goldens/user_mode/fly_obb_tracks.csv")
    assert set(CORE_COLUMNS).issubset(golden.columns)


def test_ant_cnn_identity_has_real_identity_values():
    golden = pd.read_csv("tests/goldens/user_mode/ant_cnn_identity_tracks.csv")
    assert set(IDENTITY_COLUMNS).issubset(golden.columns)
    # Real CNN identities resolved (not the "unknown" placeholder every row).
    assert (golden["identity"] != "unknown").any()


def test_pose_fixture_has_keypoint_triples():
    golden = pd.read_csv("tests/goldens/user_mode/ant_pose_headtail_tracks.csv")
    assert set(CORE_COLUMNS).issubset(golden.columns)
    assert any(c.endswith("_conf") for c in golden.columns)
    assert any(c.endswith("_x") for c in golden.columns)
    assert any(c.endswith("_y") for c in golden.columns)
    assert "heading_deg" in golden.columns
