"""End-to-end execution of the autotune sidecar child process.

No other test in the suite executes ``sidecar_child``; this file is the
regression anchor for every finding that only manifests when the child
actually runs. It is expected to be RED today: ``make_roi_params`` (see
``tests/autotune_helpers.py``) builds engine params with a whole-frame ROI,
which makes ``build_engine_params`` emit ``ARENA_LABELS`` (a uint16
ndarray) alongside ``ROI_MASK``. ``write_sidecar_request`` only special-cases
``ROI_MASK`` when staging numpy arrays out of the JSON payload
(``sidecar.py::_json_value``), so ``ARENA_LABELS`` blows up with a
``TypeError`` before the child is ever spawned. Finding B1; fixed in Task 2.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hydra_suite.core.inference.autotune.sidecar import (
    SidecarTrialSpec,
    write_sidecar_request,
)

from .autotune_helpers import _resource_probe, _settings, make_roi_params

FIXTURES = Path(__file__).resolve().parents[1] / "tools/equivalence/fixtures/clips"
CLIP = FIXTURES / "fly_obb.mp4"
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

pytestmark = pytest.mark.sidecar_e2e


@pytest.mark.skipif(not CLIP.exists(), reason="equivalence fixtures not fetched")
def test_sidecar_child_completes_on_a_project_with_an_roi(tmp_path):
    params = make_roi_params(CLIP)
    # A malformed ROI shape rasterizes to an all-zero ARENA_LABELS (every
    # pixel gated out as outside-ROI), which fails ~400 lines downstream
    # with an opaque "calibration tracking pass produced no rows". Assert
    # the fixture's ROI actually rasterized to something non-degenerate
    # BEFORE the child is launched, so a future regression here fails loud
    # and immediately.
    arena_labels = params.get("ARENA_LABELS")
    assert arena_labels is not None, "expected ARENA_LABELS in ROI fixture params"
    assert arena_labels.any(), "ARENA_LABELS is all-zero -- ROI rasterized to nothing"
    observation, probe = _resource_probe()
    spec = SidecarTrialSpec(
        video_path=CLIP,
        params=params,
        observation=observation,
        resource_probe=probe,
        start_frame=0,
        # A PRODUCTION-SIZED window. maximum_frames is deliberately left at
        # its default (640, divided across five blocks by _frames_for_block),
        # so this block measures 128 frames -- the same size a real
        # calibration uses, and long enough that the fixture's configured
        # MIN_TRAJECTORY_LENGTH (min_trajectory_length_seconds 0.33 * 100 fps
        # = 33) applies unclamped. A short window here would have made the
        # test green only because postprocessing had been relaxed.
        end_frame=499,
    )

    # This is the exact call ``ContainedTrialExecutor._run_once`` makes: a
    # ``SidecarTrialSpec`` + settings staged via ``write_sidecar_request``,
    # producing a request.json under an uncreated request root.
    request = write_sidecar_request(
        tmp_path / "ipc",
        spec,
        _settings(),
        phase="full",
        field_name=None,
        block_index=0,
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "hydra_suite.core.inference.autotune.sidecar_child",
            "--request",
            str(request),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )

    assert proc.returncode == 0, f"child failed:\n{proc.stderr[-4000:]}"
    result_path = request.parent / "output" / "result.json"
    result = json.loads(result_path.read_text())
    assert result["measured_frames"] > 0
    forward = request.parent / str(result["forward_csv"])
    assert forward.exists() and len(forward.read_text().splitlines()) > 1
