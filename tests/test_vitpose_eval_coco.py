import os
from pathlib import Path

import pytest
import torch

ASSET_DIR = Path(os.path.expanduser("~/.cache/vitpose-assets"))

requires_coco = pytest.mark.skipif(
    not (ASSET_DIR / "val2017").exists()
    or not (ASSET_DIR / "COCO_val2017_detections_AP_H_56_person.json").exists(),
    reason="COCO val2017 + detections required; see Task 10 Step 1",
)


def _device() -> str:
    """Pick the best available device.

    The gates must run on whatever box they land on -- hardcoding "mps" made
    them raise `PyTorch is not linked with support for mps devices` on the CUDA
    box, defeating the whole point of validating on the deployment target. This
    also honours the plan's "never hardcode a device" constraint.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@requires_coco
def test_smoke_eval_on_20_images():
    """Fast feedback before the full ~40min run.

    Asserts AP > 0.5, not just 0<=AP<=1 (which is vacuously true for any
    result, including a totally broken pipeline). A correctly-ported ViTPose-B
    scores ~0.76 on full val; 20 images is noisy but a working pipeline clears
    0.5 comfortably, while a broken one lands near 0.
    """
    from tools.vitpose.eval_coco import evaluate

    res = evaluate("B", "classic", ASSET_DIR / "vitpose-b.pth", _device(), limit=20)
    assert res["AP"] > 0.5, f"smoke AP {res['AP']:.3f} — pipeline is broken"


@requires_coco
def test_forward_fn_injection_actually_drives_the_pipeline():
    """The Task 13 Step 5 TensorRT eval routes forward passes through
    ``forward_fn`` instead of the torch model. Guard against the two ways that
    injection could be silently wrong:

    1. A real bug where ``evaluate`` still builds/loads the default torch
       model even when ``forward_fn`` is given (a "silent fallback") -- caught
       here by pointing ``ckpt`` at a path that does not exist. If the
       fallback ever fires, ``load_checkpoint`` raises before this test's
       assertions even run.
    2. ``forward_fn`` being accepted but never actually called for both the
       normal and flipped passes -- caught by counting calls on a wrapper
       around the real model and checking against AP computed the ordinary
       (non-injected) way on the same 20 images. If the injected path skipped
       the flip test, or fell back to zeros/identity, the AP would diverge
       from the baseline.
    """
    from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose
    from hydra_suite.core.identity.pose.vitpose.weights import load_checkpoint
    from tools.vitpose.eval_coco import evaluate

    device = _device()
    baseline = evaluate("B", "classic", ASSET_DIR / "vitpose-b.pth", device, limit=20)

    model = build_vitpose("B", "classic").eval().to(device)
    load_checkpoint(model, ASSET_DIR / "vitpose-b.pth", strict=True)

    calls = {"n": 0}

    def counting_forward(x: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        with torch.no_grad():
            return model(x)

    injected = evaluate(
        "B",
        "classic",
        ASSET_DIR / "this-checkpoint-does-not-exist.pth",
        device,
        limit=20,
        forward_fn=counting_forward,
    )

    assert calls["n"] > 0, "forward_fn was never invoked -- injection is dead code"
    assert calls["n"] % 2 == 0, (
        "forward_fn must be called in pairs (normal + flipped) per batch, "
        f"got {calls['n']} calls"
    )
    assert injected["AP"] == pytest.approx(baseline["AP"], abs=1e-9), (
        f"injected forward_fn produced AP={injected['AP']:.4f} vs baseline "
        f"{baseline['AP']:.4f} -- the injected path is not equivalent to the "
        "default torch path"
    )


@pytest.mark.coco_eval
@requires_coco
def test_gate_c_classic_reproduces_published_ap():
    """GATE C. Published ViTPose-B classic = 75.8 AP.

    Diagnostic ladder if this fails:
      ~1 AP off    -> UDP mismatch (warp and decode disagree)
      ~0.3 AP off  -> decode blur sigma
      wildly off   -> patch padding or pos-embed
    """
    from tools.vitpose.eval_coco import evaluate

    res = evaluate("B", "classic", ASSET_DIR / "vitpose-b.pth", _device())
    assert abs(res["AP"] * 100 - 75.8) < 0.2, f"got {res['AP'] * 100:.2f} AP"


@pytest.mark.coco_eval
@requires_coco
def test_gate_c_simple_reproduces_published_ap():
    """GATE C. Published ViTPose-B simple = 75.5 AP."""
    from tools.vitpose.eval_coco import evaluate

    res = evaluate("B", "simple", ASSET_DIR / "vitpose-b-simple.pth", _device())
    assert abs(res["AP"] * 100 - 75.5) < 0.2, f"got {res['AP'] * 100:.2f} AP"
