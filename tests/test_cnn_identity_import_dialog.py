"""Non-Qt logic in CNNIdentityImportDialog: metadata summary + scoring mode persistence."""

from __future__ import annotations


def test_describe_cnn_identity_candidate_flat(tiny_flat_subset):
    from hydra_suite.trackerkit.gui.dialogs.cnn_identity_import_dialog import (
        describe_cnn_identity_candidate,
    )

    summary = describe_cnn_identity_candidate(str(tiny_flat_subset))
    assert summary["arch"] == "tinyclassifier"
    assert summary["is_multihead"] is False
    assert summary["factor_names"] == ["flat"]
    assert summary["class_names_per_factor"] == [["left", "right"]]
    assert summary["input_size"] == (64, 64)


def test_describe_cnn_identity_candidate_multihead(tiny_multi_identity):
    from hydra_suite.trackerkit.gui.dialogs.cnn_identity_import_dialog import (
        describe_cnn_identity_candidate,
    )

    summary = describe_cnn_identity_candidate(str(tiny_multi_identity))
    assert summary["is_multihead"] is True
    assert summary["factor_names"] == ["color", "shape"]
    assert summary["class_names_per_factor"] == [["r", "g", "b"], ["sq", "ci"]]
