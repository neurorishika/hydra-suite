import os
from pathlib import Path

import pytest

ASSET_DIR = Path(os.path.expanduser("~/.cache/vitpose-assets"))

requires_coco = pytest.mark.skipif(
    not (ASSET_DIR / "val2017").exists()
    or not (ASSET_DIR / "COCO_val2017_detections_AP_H_56_person.json").exists(),
    reason="COCO val2017 + detections required; see Task 10 Step 1",
)


@requires_coco
def test_smoke_eval_on_20_images():
    """Fast feedback before the full ~40min run.

    Asserts AP > 0.5, not just 0<=AP<=1 (which is vacuously true for any
    result, including a totally broken pipeline). A correctly-ported ViTPose-B
    scores ~0.76 on full val; 20 images is noisy but a working pipeline clears
    0.5 comfortably, while a broken one lands near 0.
    """
    from tools.vitpose.eval_coco import evaluate

    res = evaluate("B", "classic", ASSET_DIR / "vitpose-b.pth", "mps", limit=20)
    assert res["AP"] > 0.5, f"smoke AP {res['AP']:.3f} — pipeline is broken"


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

    res = evaluate("B", "classic", ASSET_DIR / "vitpose-b.pth", "mps")
    assert abs(res["AP"] * 100 - 75.8) < 0.2, f"got {res['AP'] * 100:.2f} AP"


@pytest.mark.coco_eval
@requires_coco
def test_gate_c_simple_reproduces_published_ap():
    """GATE C. Published ViTPose-B simple = 75.5 AP."""
    from tools.vitpose.eval_coco import evaluate

    res = evaluate("B", "simple", ASSET_DIR / "vitpose-b-simple.pth", "mps")
    assert abs(res["AP"] * 100 - 75.5) < 0.2, f"got {res['AP'] * 100:.2f} AP"
