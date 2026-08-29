from pathlib import Path

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
    monkeypatch.setattr(ck, "_find_spec", lambda name: object())
    monkeypatch.setattr(ck, "_has_predictor_symbol", lambda: True)
    avail = ck.probe_availability(cache_dir=tmp_path)
    assert avail.usable is False
    assert "checkpoint" in avail.reason.lower()
    # C1: the STRUCTURED distinction. A missing checkpoint is not the same
    # kind of unavailable as a missing dependency: the download offer lives
    # inside the dialog behind the button, so gating the button on
    # `usable` alone made the whole feature unreachable.
    assert avail.checkpoint_missing is True
    assert avail.actionable is True


def test_probe_reports_a_missing_python_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ck, "_find_spec", lambda name: None if name == "ftfy" else object()
    )
    avail = ck.probe_availability(cache_dir=tmp_path)
    assert avail.usable is False
    assert "ftfy" in avail.reason
    # A genuinely unusable install: NOT confusable with a missing checkpoint,
    # so the GUI keeps the button disabled here.
    assert avail.checkpoint_missing is False
    assert avail.actionable is False


def test_probe_succeeds_when_everything_is_present(tmp_path, monkeypatch):
    monkeypatch.setattr(ck, "_find_spec", lambda name: object())
    monkeypatch.setattr(ck, "_has_predictor_symbol", lambda: True)
    (tmp_path / f"{ck.DEFAULT_VARIANT}.pt").write_bytes(b"x")
    avail = ck.probe_availability(cache_dir=tmp_path)
    assert avail.usable is True
    assert avail.reason == ""
    assert avail.checkpoint_missing is False


def test_labeler_refuses_to_construct_when_the_probe_fails(tmp_path, monkeypatch):
    from hydra_suite.core.inference.semantic import sam3

    monkeypatch.setattr(
        sam3,
        "probe_availability",
        lambda *a, **k: ck.Sam3Availability(False, "no ftfy"),
    )
    with pytest.raises(RuntimeError, match="no ftfy"):
        sam3.Sam3SemanticLabeler.from_variant(cache_dir=tmp_path)


def test_labeler_tolerates_only_a_missing_checkpoint_not_a_missing_dep(
    tmp_path, monkeypatch
):
    """C1 / deferred-minor-3: the from_variant guard keys on the STRUCTURED
    field, not on the substring "not downloaded" in a human-readable reason.

    The old guard read `"not downloaded" not in reason`, so rewording the
    probe's message -- which C1 does -- would have silently turned the
    tolerated case into a hard failure.
    """
    from hydra_suite.core.inference.semantic import sam3

    calls = []
    monkeypatch.setattr(
        sam3,
        "ensure_checkpoint",
        lambda *a, **k: (calls.append(1), tmp_path / "sam3.pt")[1],
    )
    monkeypatch.setattr(
        sam3,
        "probe_availability",
        # A reason that contains NEITHER "not downloaded" nor "unavailable".
        lambda *a, **k: ck.Sam3Availability(
            False, "the weights are absent", checkpoint_missing=True
        ),
    )
    # Must get PAST the guard. Whatever happens afterwards (the real
    # ultralytics import/ctor) is not this test's business -- what matters is
    # that no RuntimeError names the probe's reason, and that
    # ensure_checkpoint was reached.
    try:
        sam3.Sam3SemanticLabeler.from_variant(cache_dir=tmp_path)
    except Exception as exc:  # pragma: no cover - depends on optional assets
        assert "the weights are absent" not in str(exc)
    assert calls, "ensure_checkpoint must be reached for a missing checkpoint"


def test_ensure_checkpoint_never_reads_the_whole_file_into_memory(
    tmp_path, monkeypatch
):
    """Minor: `dest.write_bytes(src.read_bytes())` peaked at 3.45 GB RSS."""
    src = tmp_path / "hf" / "sam3.pt"
    src.parent.mkdir()
    src.write_bytes(b"weights")

    def _no_read_bytes(self, *a, **k):  # pragma: no cover - failure path
        raise AssertionError("must not slurp the checkpoint into memory")

    monkeypatch.setattr(ck, "hf_hub_download", lambda **k: str(src))
    monkeypatch.setattr(Path, "read_bytes", _no_read_bytes)
    dest = ck.ensure_checkpoint(cache_dir=tmp_path / "cache")
    assert dest.exists()
    assert dest.stat().st_size == len(b"weights")


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


def test_missing_clip_names_an_install_that_can_actually_fix_it(tmp_path, monkeypatch):
    """F: `clip` is NOT in the sam3 extra and cannot be.

    It is a PEP 508 direct reference, which PyPI rejects in uploaded
    metadata, so it was deliberately dropped from the extra -- leaving the
    probe telling the user to run an install that could never satisfy it.
    """
    monkeypatch.setattr(
        ck, "_find_spec", lambda name: None if name == "clip" else object()
    )
    avail = ck.probe_availability(cache_dir=tmp_path)
    assert avail.usable is False
    assert "clip" in avail.reason
    assert "github.com/openai/CLIP.git" in avail.reason
    assert "hydra-suite[sam3]" not in avail.reason
    # The other deps DO come from the extra.
    monkeypatch.setattr(
        ck, "_find_spec", lambda name: None if name == "ftfy" else object()
    )
    assert "hydra-suite[sam3]" in ck.probe_availability(cache_dir=tmp_path).reason
