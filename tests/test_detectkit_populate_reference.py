from hydra_suite.detectkit.gui.models import (
    SliceTrainingSettings,
    populate_measured_reference,
)


def test_populate_sets_when_unset():
    s = SliceTrainingSettings(reference_body_px=0.0)
    changed = populate_measured_reference(s, 55.0)
    assert changed is True
    assert s.reference_body_px == 55.0


def test_populate_preserves_user_value():
    s = SliceTrainingSettings(reference_body_px=30.0)
    changed = populate_measured_reference(s, 55.0)
    assert changed is False
    assert s.reference_body_px == 30.0


def test_populate_ignores_zero_measured():
    s = SliceTrainingSettings(reference_body_px=0.0)
    changed = populate_measured_reference(s, 0.0)
    assert changed is False
    assert s.reference_body_px == 0.0
