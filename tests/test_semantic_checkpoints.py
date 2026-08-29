import numpy as np
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


def test_labeler_refuses_to_construct_when_the_probe_fails(tmp_path, monkeypatch):
    from hydra_suite.core.inference.semantic import sam3

    monkeypatch.setattr(sam3, "probe_availability", lambda *a, **k: (False, "no ftfy"))
    with pytest.raises(RuntimeError, match="no ftfy"):
        sam3.Sam3SemanticLabeler.from_variant(cache_dir=tmp_path)


def test_labeler_satisfies_the_protocol_without_weights():
    from hydra_suite.core.inference.semantic.base import SemanticLabeler
    from hydra_suite.core.inference.semantic.sam3 import Sam3SemanticLabeler

    stub = Sam3SemanticLabeler(predictor=object(), device="cpu")
    assert isinstance(stub, SemanticLabeler)
    assert stub.name == "sam3"


def test_label_image_calls_the_predictor_with_a_text_list_prompt():
    """Pins the `text=[prompt]` contract.

    ultralytics' predictor.__call__ forwards unmatched kwargs into
    SAM3SemanticPredictor.inference()'s **kwargs sink and silently drops
    them: the concept-prompt keyword there is `text` (a list[str]), never
    `prompt`. A regression back to `prompt=prompt` would make every call
    run promptless, and predict.py:2288 does `len(text)` on it, so passing
    a bare string (not wrapped in a list) would also be silently
    misread as a one-char-per-class prompt. This test fails on either
    regression without needing any real weights.
    """
    from hydra_suite.core.inference.semantic.sam3 import Sam3SemanticLabeler

    class FakePredictor:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return []

    fake = FakePredictor()
    labeler = Sam3SemanticLabeler(predictor=fake, device="cpu")
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    labeler.label_image(image, "ant")

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert "text" in call, "must call the predictor with the `text` keyword"
    assert "prompt" not in call, "`prompt` is silently dropped by ultralytics"
    assert isinstance(call["text"], list), "must be a list, not a bare string"
    assert call["text"] == ["ant"]
