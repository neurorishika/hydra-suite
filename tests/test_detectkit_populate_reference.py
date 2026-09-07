from hydra_suite.detectkit.gui.models import (
    SliceTrainingSettings,
    populate_measured_reference,
)


def test_populate_sets_when_unset():
    s = SliceTrainingSettings(reference_body_px=0.0)
    changed = populate_measured_reference(s, 55.0)
    assert changed is True
    assert s.reference_body_px == 55.0


def test_populate_refreshes_a_stale_measurement():
    """``reference_body_px`` is label-derived metadata, not a user override.

    ``populate_measured_reference`` deliberately re-measures it on every sliced
    dataset build, so a differing stored value is replaced (and reported as
    changed) rather than preserved.
    """
    s = SliceTrainingSettings(reference_body_px=30.0)
    changed = populate_measured_reference(s, 55.0)
    assert changed is True
    assert s.reference_body_px == 55.0


def test_populate_ignores_zero_measured():
    s = SliceTrainingSettings(reference_body_px=0.0)
    changed = populate_measured_reference(s, 0.0)
    assert changed is False
    assert s.reference_body_px == 0.0
