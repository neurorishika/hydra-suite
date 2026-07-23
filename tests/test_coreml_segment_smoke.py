"""CoreML segment-as-OBB export+reload smoke test.

Closes the one acceleration cell an independent runtime review flagged as
UNTESTED for the segment-as-OBB feature: CoreML (Apple ``gpu_fast`` tier) x
``model_task="segment"``. CoreML loads a YOLO model as a plain ultralytics
model (task inferred on reload from the ``.mlpackage``'s own metadata, not
passed explicitly -- see ``_load_coreml_executor`` in
``runtime_artifacts.py``), and OBB extraction from the segment masks is
dispatched downstream via ``_extract_obb_from_masks``
(``core/inference/stages/obb.py``). This test exercises the real export +
reload + extraction chain end to end through the SAME entry points the
production pipeline uses -- ``load_obb_models`` / ``run_obb`` -- rather than a
bespoke export, to confirm the routing that was previously only reasoned
about actually produces a structurally valid ``OBBResult``.

Guarded to never break CI or non-Apple machines: skips (does not fail/error)
unless running on macOS/arm64 with ``coremltools`` importable and a
pretrained ``yolo11n-seg.pt`` checkpoint can actually be downloaded. This
repo's only registered pytest marker is ``benchmark`` (excluded from the
default run via ``addopts = -m "not benchmark"`` in ``pytest.ini``); the
existing hardware-gated real-export smoke tests
(``tests/test_obb_coreml_export.py::test_coreml_real_export_smoke``,
``tests/test_coreml_determinism.py``) do not use it and instead rely purely
on ``skipif``/``importorskip`` guards, so this test follows that same
convention rather than introducing an unregistered ``slow``/``network``
marker.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys

import numpy as np
import pytest


def _require_apple_silicon() -> None:
    if sys.platform != "darwin":
        pytest.skip("CoreML segment smoke test requires macOS")
    if platform.machine() not in ("arm64", "aarch64"):
        pytest.skip("CoreML segment smoke test requires Apple Silicon (arm64)")


def test_coreml_segment_obb_smoke(tmp_path):
    """Real CoreML export+reload of a segment checkpoint, run through the
    production ``load_obb_models``/``run_obb`` entry points, must yield a
    structurally valid ``OBBResult`` -- zero detections on a synthetic frame
    is a pass; a crash or malformed result is not.
    """
    _require_apple_silicon()
    pytest.importorskip("coremltools")

    try:
        from ultralytics import YOLO
    except ImportError:
        pytest.skip("ultralytics not installed")

    # --- 1. Obtain a standard pretrained segmentation checkpoint ----------
    # ultralytics auto-downloads on first use; run inside tmp_path so the
    # download (and later the .mlpackage export) never touches the repo or
    # gets committed. Any network/download failure is a skip, not a failure --
    # this test verifies OUR extraction code, not network availability.
    pt_name = "yolo11n-seg.pt"
    orig_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        try:
            base_model = YOLO(pt_name)
        except Exception as exc:  # network unavailable, download failed, etc.
            pytest.skip(f"Could not obtain pretrained {pt_name}: {exc}")
        # Resolve wherever ultralytics actually placed/cached the checkpoint
        # to a concrete local path we control and can hand to our own config.
        ckpt_path = getattr(base_model, "ckpt_path", None)
        if not ckpt_path or not os.path.exists(ckpt_path):
            candidate = tmp_path / pt_name
            if not candidate.exists():
                pytest.skip(
                    f"Pretrained checkpoint {pt_name} not found on disk after load"
                )
            ckpt_path = str(candidate)
        local_pt = tmp_path / pt_name
        if os.path.abspath(ckpt_path) != os.path.abspath(local_pt):
            shutil.copy(ckpt_path, local_pt)
        del base_model
    finally:
        os.chdir(orig_dir)

    # --- 2. Load through the REAL pipeline entry points --------------------
    from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
    from hydra_suite.core.inference.result import OBBResult
    from hydra_suite.core.inference.runtime import RuntimeContext
    from hydra_suite.core.inference.runtime_artifacts import ArtifactExportError
    from hydra_suite.core.inference.stages.obb import load_obb_models, run_obb

    config = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path=str(local_pt), model_task="segment"),
    )
    # coreml_mode=True mirrors what RuntimeContext.from_config produces for the
    # Apple gpu_fast tier once the resolver selects the CoreML backend (see
    # runtime.py: coreml_mode = resolved.backend == "coreml"); constructed
    # directly here so the test is deterministic and doesn't depend on the
    # resolver's artifact-availability heuristics for its skip/run decision.
    runtime = RuntimeContext(
        cuda_mode=False,
        device="mps",
        use_nvdec=False,
        tensor_on_cuda=False,
        coreml_mode=True,
        requested_gpu=True,
    )
    # runtime_to_compute_runtime(runtime) == "coreml" for this RuntimeContext
    # -> load_obb_models routes to load_obb_executor(..., "coreml", ...) ->
    # _load_coreml_executor, the real export+reload path (auto_export=True by
    # OBBDirectConfig's default), exporting into tmp_path next to the .pt.
    os.chdir(tmp_path)
    try:
        try:
            models = load_obb_models(config, runtime, batch_size=1)
        except ArtifactExportError as exc:
            pytest.skip(f"CoreML export unavailable in this environment: {exc}")
        except (ImportError, ModuleNotFoundError) as exc:
            # Missing coremltools (or a dependency it pulls in, e.g. protobuf)
            # is a genuinely environmental condition -- skip.
            pytest.skip(f"CoreML export dependency unavailable: {exc}")
        except RuntimeError as exc:
            # coremltools/xcoreml raise RuntimeError for a missing Xcode
            # command-line-tools toolchain (the actual export/compile step
            # shells out to `xcrun`). Narrowly matched on that specific,
            # well-known environmental failure mode; anything else re-raises
            # so a real export regression in OUR code is not silently
            # swallowed.
            msg = str(exc).lower()
            if "xcrun" in msg or "xcode" in msg:
                pytest.skip(f"CoreML export requires Xcode command-line tools: {exc}")
            raise

        assert models.mode == "direct"
        assert models.direct_model is not None

        # --- 3. Run ONE synthetic frame through the real inference path ----
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        outputs = run_obb([frame], models, config, runtime)
    finally:
        os.chdir(orig_dir)

    # --- 4. Assert the returned OBBResult is structurally valid ------------
    assert len(outputs) == 1
    result = outputs[0]
    assert isinstance(
        result, OBBResult
    ), f"Expected OBBResult from the coreml+segment path, got {type(result)}"

    n = result.num_detections
    assert result.centroids.shape == (n, 2)
    assert result.angles.shape == (n,)
    assert result.sizes.shape == (n,)
    assert result.shapes.shape == (n, 2)
    assert result.confidences.shape == (n,)
    assert result.corners.shape == (n, 4, 2)
    assert result.detection_ids.shape == (n,)
    assert result.frame_idx == 0

    if n > 0:
        assert np.all(np.isfinite(result.centroids))
        assert np.all(np.isfinite(result.angles))
        assert np.all(np.isfinite(result.sizes))
        assert np.all(np.isfinite(result.shapes))
        assert np.all(np.isfinite(result.corners))
        # Angles are folded to [0, pi) by _normalize_obb_geometry.
        assert np.all(result.angles >= 0.0)
        assert np.all(result.angles < np.pi + 1e-4)
        assert np.all(result.sizes > 0.0)
        assert np.all((result.confidences >= 0.0) & (result.confidences <= 1.0))
    else:
        # n == 0 on random noise input is expected: the point is that the
        # CoreML-loaded segment model + mask-to-OBB extraction ran to
        # completion and produced a *well-formed* empty result, not that it
        # is unconditionally acceptable -- assert the empty-case field
        # shapes/dtypes explicitly rather than skipping all assertions.
        assert result.centroids.shape == (0, 2)
        assert result.centroids.dtype == np.float32
        assert result.angles.shape == (0,)
        assert result.angles.dtype == np.float32
        assert result.sizes.shape == (0,)
        assert result.sizes.dtype == np.float32
        assert result.shapes.shape == (0, 2)
        assert result.shapes.dtype == np.float32
        assert result.confidences.shape == (0,)
        assert result.confidences.dtype == np.float32
        assert result.corners.shape == (0, 4, 2)
        assert result.corners.dtype == np.float32
        assert result.detection_ids.shape == (0,)
        assert result.detection_ids.dtype == np.int64
