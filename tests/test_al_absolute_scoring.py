from hydra_suite.utils.geometry import obb_corners_from_dims


def _box(cx, cy, w=40.0, h=16.0, theta=0.0):
    return obb_corners_from_dims(cx, cy, w, h, theta)


def test_fragmentation_is_zero_for_well_separated_normal_boxes():
    from hydra_suite.data.al.signals import score_fragmentation

    boxes = [_box(100, 100), _box(300, 300), _box(500, 100)]
    assert score_fragmentation(boxes) == 0.0


def test_fragmentation_fires_on_two_small_overlapping_boxes():
    """One animal split into two half-size detections sitting on top of it."""
    from hydra_suite.data.al.signals import score_fragmentation

    boxes = [
        _box(300, 300),  # normal-size animal
        _box(500, 300),  # normal-size animal
        _box(100, 100, w=18.0, h=8.0),  # fragment
        _box(108, 102, w=18.0, h=8.0),  # its twin, close + small
    ]
    assert score_fragmentation(boxes) > 0.45


def test_fragmentation_needs_at_least_two_boxes():
    from hydra_suite.data.al.signals import score_fragmentation

    assert score_fragmentation([]) == 0.0
    assert score_fragmentation([_box(100, 100)]) == 0.0


def test_fragmentation_is_bounded():
    from hydra_suite.data.al.signals import score_fragmentation

    boxes = [_box(100, 100, w=5, h=3), _box(100, 100, w=5, h=3)]
    assert 0.0 <= score_fragmentation(boxes) <= 1.0


def test_tracker_default_preset_weights_fragmentation():
    from hydra_suite.data.al.acquisition import PRESETS

    assert PRESETS["tracker_default"].fragmentation == 0.30


def test_al_signals_carries_fragmentation_field():
    from hydra_suite.data.al.signals import ALSignals

    assert ALSignals(frame_id=0).fragmentation_score == 0.0
