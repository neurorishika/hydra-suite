from hydra_suite.core.post.interpolated_crops import run_interpolated_crops


def test_missing_csv_returns_empty_payload(tmp_path):
    result = run_interpolated_crops(
        str(tmp_path / "nope.csv"),
        str(tmp_path / "nope.mp4"),
        str(tmp_path / "nope.npz"),
        {},
    )
    # _validate_and_setup returns None on a missing CSV; the pipeline yields the
    # documented "nothing produced" payload rather than raising.
    assert isinstance(result, dict)
    assert result.get("saved", 0) == 0


def test_should_stop_before_setup_returns_empty_payload(tmp_path):
    result = run_interpolated_crops(
        str(tmp_path / "any.csv"),
        str(tmp_path / "any.mp4"),
        str(tmp_path / "any.npz"),
        {},
        should_stop=lambda: True,
    )
    assert result.get("saved", 0) == 0
