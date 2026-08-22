"""Golden span-path set: the durable guard against a silently-dropped span."""

import json
from pathlib import Path

from hydra_suite.core.tracking.profiler import TrackingProfiler
from hydra_suite.utils import profiling_names as N
from hydra_suite.utils.profiling import span

GOLDEN = Path(__file__).parent / "span_golden_paths.json"


def _paths(node, prefix=()):
    """Flatten a snapshot to the set of slash-joined span paths."""
    out = set()
    for child in node["children"]:
        path = prefix + (child["name"],)
        out.add("/".join(path))
        out |= _paths(child, path)
    return out


def _synthetic_batch_window():
    """Mirror the real nesting of one batch window without loading models.

    Every `with span(...)` here must correspond to a real placement in
    pipeline.py / crops.py / the stage modules. When the two drift, this test
    fails -- which is the entire point.
    """
    with span(N.INFERENCE), span(N.BATCH_PASS):
        with span(N.OPEN_CACHES):
            pass
        with span(N.WINDOW, units=1):
            with span(N.DETECT, units=1):
                with span(N.RUN_OBB, units=1):
                    with span(N.MODEL_EXECUTE, gpu=True):
                        pass
                    with span(N.EXTRACT_RAW):
                        pass
            with span(N.MATERIALIZE, units=1):
                pass
            for stage in (N.HEADTAIL, N.CNN):
                with span(stage, units=4):
                    with span(N.CROP_EXTRACT):
                        with span(N.AFFINE_LOOP):
                            pass
                        with span(N.WARP_BATCH, units=4, gpu=True):
                            with span(N.FRAME_TO_CHW, units=4):
                                pass
                    with span(N.APPLY_FIT, units=4):
                        pass
                    with span(N.BACKEND_FORWARD, units=4, gpu=True):
                        pass
            # pose's stage-level structure diverges from headtail/cnn: the
            # real run_pose_batch (stages/pose.py) wraps the per-crop prep
            # loop (affine inversion + apply_fit) in PREP_LOOP, then runs
            # BACKEND_FORWARD as a sibling -- there is no crop_extract span
            # and apply_fit is called undecorated inside the prep loop, so no
            # APPLY_FIT span exists under pose either.
            with span(N.POSE, units=4):
                with span(N.PREP_LOOP, units=4):
                    pass
                with span(N.BACKEND_FORWARD, units=4, gpu=True):
                    pass
            with span(N.CACHE_WRITE):
                with span(N.ENQUEUE):
                    pass
            with span(N.ASSEMBLE_SCATTER):
                pass


def test_golden_span_paths():
    prof = TrackingProfiler(enabled=True)
    with prof.armed():
        _synthetic_batch_window()
    actual = _paths(prof.spans.snapshot())
    expected = set(json.loads(GOLDEN.read_text()))

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    assert not missing, (
        f"span paths disappeared: {missing}\n"
        "A refactor dropped a span. Restore it, or update the golden set "
        "DELIBERATELY with a note in the commit message."
    )
    assert not extra, f"new span paths not in the golden set: {extra}"


def test_crop_extract_and_backend_forward_are_siblings():
    """Regression guard for the head-tail/CNN blending defect.

    24.0s of the 34.4s originating defect lived in the head-tail + CNN crop
    path. If crop_extract nests UNDER backend_forward, that cost blends with
    model time in one self_s and the tree indicts the wrong function.
    """
    prof = TrackingProfiler(enabled=True)
    with prof.armed():
        _synthetic_batch_window()
    paths = _paths(prof.spans.snapshot())
    for stage in ("headtail", "cnn"):
        base = f"inference/batch_pass/window/{stage}"
        assert f"{base}/crop_extract" in paths
        assert f"{base}/backend_forward" in paths
        assert f"{base}/backend_forward/crop_extract" not in paths

    # pose has no crop_extract span (see _synthetic_batch_window): its
    # analogous guard is that prep_loop and backend_forward are siblings,
    # not that prep_loop nests under backend_forward.
    pose_base = "inference/batch_pass/window/pose"
    assert f"{pose_base}/prep_loop" in paths
    assert f"{pose_base}/backend_forward" in paths
    assert f"{pose_base}/backend_forward/prep_loop" not in paths
