import pytest

from hydra_suite.utils.geometry_levels import GeometryLevel

pytest.importorskip("PySide6")


@pytest.mark.parametrize(
    "params,expected_enabled",
    [
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "segment"},
            {"polygon", "obb", "aabb"},
        ),
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "obb"},
            {"obb", "aabb"},
        ),
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "detect"},
            {"aabb"},
        ),
    ],
)
def test_level_checkboxes_reflect_detector_capability(params, expected_enabled):
    """Pure logic: which level checkboxes a given detector config enables."""
    from hydra_suite.data.al.escalation import achievable_levels
    from hydra_suite.data.dataset_generation import resolve_native_level

    enabled = {lvl.label for lvl in achievable_levels(resolve_native_level(params))}
    assert enabled == expected_enabled


def test_level_status_text_names_the_missing_requirement():
    from hydra_suite.trackerkit.gui.panels.dataset_panel import format_level_status

    text = format_level_status(GeometryLevel.OBB)
    assert "obb" in text and "aabb" in text
    assert "segmentation" in text.lower()

    assert "polygon" in format_level_status(GeometryLevel.POLYGON)
