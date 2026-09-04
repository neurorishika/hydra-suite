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


def test_label_image_applies_instance_cap_before_predictor_materializes_masks():
    from types import SimpleNamespace

    from hydra_suite.core.inference.semantic.sam3 import Sam3SemanticLabeler

    class FakePredictor:
        def __init__(self):
            self.args = SimpleNamespace(max_det=300)
            self.seen_max_det = None

        def __call__(self, **kwargs):
            self.seen_max_det = self.args.max_det
            return []

    fake = FakePredictor()
    labeler = Sam3SemanticLabeler(predictor=fake, device="cpu")
    labeler.label_image(np.zeros((4, 4, 3), dtype=np.uint8), "ant", max_instances=17)

    assert fake.seen_max_det == 17
    assert fake.args.max_det == 300, "the shared predictor setting must be restored"


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
    assert "github.com/ultralytics/CLIP.git" in avail.reason
    assert "hydra-suite[sam3]" not in avail.reason
    # The other deps DO come from the extra.
    monkeypatch.setattr(
        ck, "_find_spec", lambda name: None if name == "ftfy" else object()
    )
    assert "hydra-suite[sam3]" in ck.probe_availability(cache_dir=tmp_path).reason


def test_predictor_overrides_pin_the_requested_confidence_floor():
    """F2: ultralytics filters at args.conf BEFORE our wrapper ever runs.

    BasePredictor.__init__ sets ``args.conf = 0.25`` when it is None and
    ``postprocess`` keeps only ``pred_scores > args.conf``. Overriding
    {model, device, save, verbose} but not ``conf`` therefore made the whole
    cache-floor design fiction: nothing below 0.25 could ever be cached, so
    ``recorded_confidence_floor`` lied and calibration cells 0.05-0.25 were
    all the same run.

    Asserted on the override DICT, so this needs no GPU, no checkpoint and
    no ultralytics install.
    """
    from hydra_suite.core.inference.semantic import sam3

    ov = sam3.predictor_overrides("/tmp/sam3.pt", "cpu", 0.05)
    assert "conf" in ov, "the predictor's own confidence gate is unset"
    assert ov["conf"] == pytest.approx(0.05)
    # iou is pinned too rather than inherited from ultralytics' 0.7 default.
    assert ov["iou"] == pytest.approx(sam3.PREDICTOR_NMS_IOU)
    assert ov["model"] == "/tmp/sam3.pt"
    assert ov["device"] == "cpu"
    assert ov["save"] is False and ov["verbose"] is False
    # Any floor the caller asks for is honoured, not just the default.
    assert sam3.predictor_overrides("m", "cpu", 0.02)["conf"] == pytest.approx(0.02)
    # ...and clamped into [0, 1] so a bad dialog value cannot reach the predictor.
    assert sam3.predictor_overrides("m", "cpu", -1.0)["conf"] == 0.0
    assert sam3.predictor_overrides("m", "cpu", 5.0)["conf"] == 1.0


def test_from_variant_passes_the_floor_through_to_the_predictor(tmp_path, monkeypatch):
    """The plumbing, not just the helper: from_variant must USE the floor."""
    import sys
    import types

    from hydra_suite.core.inference.semantic import sam3

    monkeypatch.setattr(
        sam3, "probe_availability", lambda *a, **k: ck.Sam3Availability(True, "")
    )
    monkeypatch.setattr(sam3, "ensure_checkpoint", lambda *a, **k: tmp_path / "s.pt")
    monkeypatch.setattr(sam3, "resolve_torch_device", lambda: "cpu")

    seen = {}

    class _FakePredictor:
        def __init__(self, overrides=None):
            seen.update(overrides or {})

    mod = types.ModuleType("ultralytics.models.sam")
    mod.SAM3SemanticPredictor = _FakePredictor
    pkg = types.ModuleType("ultralytics.models")
    pkg.sam = mod
    root = types.ModuleType("ultralytics")
    root.models = pkg
    monkeypatch.setitem(sys.modules, "ultralytics", root)
    monkeypatch.setitem(sys.modules, "ultralytics.models", pkg)
    monkeypatch.setitem(sys.modules, "ultralytics.models.sam", mod)

    sam3.Sam3SemanticLabeler.from_variant(cache_dir=tmp_path, confidence_floor=0.05)
    assert seen.get("conf") == pytest.approx(0.05)


def test_workers_ask_for_the_cache_floor_not_the_ultralytics_default():
    """All semantic child operations thread a floor through one model seam."""
    import inspect

    from hydra_suite.detectkit.sidecars import operations

    src = inspect.getsource(operations)
    assert src.count("_semantic_labeler(") == 4  # definition + three operations
    helper = inspect.getsource(operations._semantic_labeler)
    assert helper.count("from_variant(") == 1
    assert helper.count("confidence_floor=") == 1


def test_a_gated_repo_becomes_actionable_guidance_not_a_bare_401(tmp_path, monkeypatch):
    """CUDA-box finding: `facebook/sam3` is licence-gated.

    The probe promises the checkpoint "will be downloaded once, with your
    confirmation". On a machine that has not accepted the licence, the real
    `hf_hub_download` raises a bare 401 `GatedRepoError` that says nothing
    about what the user must go and do. Every first-time user hits this.
    """
    import requests
    from huggingface_hub.errors import GatedRepoError

    from hydra_suite.core.inference.semantic import checkpoints as C

    def _gated(**kwargs):
        # GatedRepoError is an HfHubHTTPError: it demands a real response.
        resp = requests.Response()
        resp.status_code = 401
        raise GatedRepoError(
            "401 Client Error. Cannot access gated repo", response=resp
        )

    monkeypatch.setattr(C, "hf_hub_download", _gated)

    with pytest.raises(C.Sam3DownloadNotAuthorized) as exc:
        C.ensure_checkpoint(C.DEFAULT_VARIANT, cache_dir=tmp_path)

    msg = str(exc.value)
    assert "huggingface.co/facebook/sam3" in msg, "no link to the licence page"
    assert "hf auth login" in msg, "no instruction to authenticate"
    assert isinstance(exc.value, RuntimeError), "must stay catchable as RuntimeError"


def test_the_probe_warns_that_the_weights_are_gated(tmp_path, monkeypatch):
    """The 'we'll fetch it for you' promise must carry its precondition."""
    from hydra_suite.core.inference.semantic import checkpoints as C

    monkeypatch.setattr(C.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(C, "_ultralytics_ok", lambda: (True, ""), raising=False)

    avail = C.probe_availability(C.DEFAULT_VARIANT, cache_dir=tmp_path)
    if avail.checkpoint_missing:
        assert "licence-gated" in avail.reason or "gated" in avail.reason


def test_staging_follows_the_hf_snapshot_symlink_to_the_real_blob(
    tmp_path, monkeypatch
):
    """CUDA-box finding: staging produced a DANGLING symlink.

    `hf_hub_download` returns a path in the HF cache snapshot dir, which on
    Linux is a symlink into `../../blobs/<sha>`. Hardlinking that entry
    copied the symlink itself, so the staged checkpoint inherited a relative
    target meaningless at its new location. `.exists()` was then False, the
    probe reported "not downloaded", and the 3.3 GB download repeated on
    every run while the feature could never start.
    """
    from hydra_suite.core.inference.semantic import checkpoints as C

    # Rebuild the HF cache layout exactly: blobs/<sha> + a snapshot symlink
    # whose target is RELATIVE, as huggingface_hub writes it.
    cache = tmp_path / "hf"
    blobs, snap = cache / "blobs", cache / "snapshots" / "rev"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (blobs / "deadbeef").write_bytes(b"SAM3-WEIGHTS")
    link = snap / "sam3.pt"
    link.symlink_to(Path("..") / ".." / "blobs" / "deadbeef")
    assert link.exists(), "fixture is wrong: the snapshot link must resolve here"

    monkeypatch.setattr(C, "hf_hub_download", lambda **kw: str(link))

    dest = C.ensure_checkpoint(C.DEFAULT_VARIANT, cache_dir=tmp_path / "models")

    # The load-bearing assertion: the staged entry must be a REAL file, not a
    # copy of HF's symlink. On Linux `os.link` preserves the symlink (verified
    # on the CUDA box) and this fails without the fix; macOS resolves it, so
    # this test only bites on the platform where the bug actually occurs.
    assert not dest.is_symlink(), "staged HF's symlink instead of the blob"
    assert dest.exists(), "staged a checkpoint that cannot be opened"
    assert dest.read_bytes() == b"SAM3-WEIGHTS", "staged path is not the real blob"
    # And the probe must now agree the checkpoint is present.
    assert not C.probe_availability(
        C.DEFAULT_VARIANT, cache_dir=tmp_path / "models"
    ).checkpoint_missing, "probe still says missing -> re-downloads forever"


def test_the_wrong_clip_fork_is_caught_by_the_probe_not_by_a_crash(monkeypatch):
    """CUDA-box finding: openai/CLIP is the wrong fork for SAM3.

    ultralytics builds the text encoder with
    `clip.simple_tokenizer.SimpleTokenizer()` and then CALLS it. openai/CLIP
    ships that class with no `__call__`, so the probe said "ready", 3.45 GB
    downloaded, the model loaded, and the run died with an opaque
    `TypeError: 'SimpleTokenizer' object is not callable` deep inside the
    text encoder. The probe must refuse up front instead.
    """
    import sys
    import types

    from hydra_suite.core.inference.semantic import checkpoints as C

    def _install_clip(callable_tokenizer):
        class _Tok:
            if callable_tokenizer:

                def __call__(self, text, context_length=77):
                    return [0]

        mod = types.ModuleType("clip.simple_tokenizer")
        mod.SimpleTokenizer = _Tok
        pkg = types.ModuleType("clip")
        pkg.simple_tokenizer = mod
        monkeypatch.setitem(sys.modules, "clip", pkg)
        monkeypatch.setitem(sys.modules, "clip.simple_tokenizer", mod)

    _install_clip(callable_tokenizer=False)
    problem = C._clip_tokenizer_problem()
    assert problem, "the wrong CLIP fork was accepted"
    assert "ultralytics/CLIP.git" in problem, "no pointer to the fork that works"

    _install_clip(callable_tokenizer=True)
    assert C._clip_tokenizer_problem() == "", "the correct fork was rejected"


def test_the_install_hint_names_the_fork_that_actually_works():
    """The hint pointed at openai/CLIP, which cannot run SAM3."""
    from hydra_suite.core.inference.semantic.checkpoints import INSTALL_HINTS

    assert "ultralytics/CLIP.git" in INSTALL_HINTS["clip"]
    assert "openai/CLIP" not in INSTALL_HINTS["clip"]
