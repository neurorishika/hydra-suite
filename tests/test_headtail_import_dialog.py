"""Non-Qt logic in HeadTailImportDialog: metadata-only validation path."""

from __future__ import annotations


def test_headtail_dialog_accept_summary_from_tiny_subset(tiny_flat_subset):
    from hydra_suite.trackerkit.gui.dialogs.headtail_import_dialog import (
        describe_headtail_candidate,
    )

    summary = describe_headtail_candidate(str(tiny_flat_subset))
    assert summary["valid"] is True
    assert summary["arch"] == "tinyclassifier"
    assert summary["normalized_labels"] == ["left", "right"]
    assert summary["input_size"] == (64, 64)


def test_headtail_dialog_rejects_multi_head(tiny_multi_identity):
    from hydra_suite.trackerkit.gui.dialogs.headtail_import_dialog import (
        describe_headtail_candidate,
    )

    summary = describe_headtail_candidate(str(tiny_multi_identity))
    assert summary["valid"] is False
    assert "multi-head" in summary["reason"].lower()


def test_headtail_dialog_rejects_non_headtail_labels(torchvision_flat_identity):
    from hydra_suite.trackerkit.gui.dialogs.headtail_import_dialog import (
        describe_headtail_candidate,
    )

    summary = describe_headtail_candidate(str(torchvision_flat_identity))
    assert summary["valid"] is False
    assert (
        "subset" in summary["reason"].lower()
        or "head-tail" in summary["reason"].lower()
    )


def test_headtail_dialog_rejects_legacy_flat_checkpoint(
    legacy_torchvision_flat_headtail,
):
    from hydra_suite.trackerkit.gui.dialogs.headtail_import_dialog import (
        describe_headtail_candidate,
    )

    summary = describe_headtail_candidate(str(legacy_torchvision_flat_headtail))
    assert summary["valid"] is False
    assert (
        "re-export" in summary["reason"].lower()
        or "schema_version" in summary["reason"].lower()
    )
