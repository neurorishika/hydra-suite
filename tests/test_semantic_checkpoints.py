import pytest

from hydra_suite.core.inference.semantic import checkpoints as ck


def test_catalog_pins_repo_and_filename():
    entry = ck.SAM3_VARIANTS[ck.DEFAULT_VARIANT]
    assert entry.repo_id == "facebook/sam3"
    assert entry.filename  # a pinned filename, never inferred at runtime


def test_ensure_checkpoint_refuses_unknown_variant():
    with pytest.raises(ValueError, match="Unknown SAM3 variant"):
        ck.ensure_checkpoint("nope")


def test_ensure_checkpoint_refuses_to_download_when_offline(tmp_path):
    with pytest.raises(ValueError, match="downloads are disabled"):
        ck.ensure_checkpoint(
            ck.DEFAULT_VARIANT, allow_download=False, cache_dir=tmp_path
        )


def test_probe_reports_missing_checkpoint_without_downloading(tmp_path, monkeypatch):
    def _boom(*a, **k):  # any download attempt is a test failure
        raise AssertionError("probe must never download")

    monkeypatch.setattr(ck, "hf_hub_download", _boom)
    ok, reason = ck.probe_availability(cache_dir=tmp_path)
    assert ok is False
    assert "checkpoint" in reason.lower()


def test_probe_reports_a_missing_python_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ck, "_find_spec", lambda name: None if name == "ftfy" else object()
    )
    ok, reason = ck.probe_availability(cache_dir=tmp_path)
    assert ok is False
    assert "ftfy" in reason


def test_probe_succeeds_when_everything_is_present(tmp_path, monkeypatch):
    monkeypatch.setattr(ck, "_find_spec", lambda name: object())
    monkeypatch.setattr(ck, "_has_predictor_symbol", lambda: True)
    (tmp_path / f"{ck.DEFAULT_VARIANT}.pt").write_bytes(b"x")
    ok, reason = ck.probe_availability(cache_dir=tmp_path)
    assert ok is True
    assert reason == ""
