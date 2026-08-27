"""Every inference crop path obeys the Layer 1 + Layer 2 contract."""

import numpy as np

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.result import OBBResult


def _obb(n, rng):
    corners = []
    for i in range(n):
        major, minor = 20.0 + 10.0 * i, 8.0 + 3.0 * i
        hw, hh = major / 2, minor / 2
        base = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        corners.append((base + np.array([100.0, 100.0])).astype(np.float32))
    corners = np.stack(corners)
    return OBBResult(
        frame_idx=0,
        centroids=np.full((n, 2), 100.0, dtype=np.float32),
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.full(n, 512.0, dtype=np.float32),
        shapes=np.full((n, 2), 2.0, dtype=np.float32),
        confidences=np.full(n, 0.9, dtype=np.float32),
        corners=corners,
        detection_ids=np.arange(n, dtype=np.int64),
    )


def test_crops_are_uniform_regardless_of_animal_size():
    from hydra_suite.core.inference.runtime import RuntimeContext
    from hydra_suite.core.inference.stages.crops import extract_canonical_crops

    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    runtime = RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False)
    crops = extract_canonical_crops(frame, _obb(3, None), g, runtime)
    assert crops.shape[0] == 3
    assert crops.shape[2] == g.canvas_h
    assert crops.shape[3] == g.canvas_w


def test_cache_key_includes_the_canonical_geometry():
    from hydra_suite.core.inference.cache.keys import canonical_geometry_key

    a = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    b = CanonicalGeometry.from_reference(20.0, 2.44, 1.6)
    assert canonical_geometry_key(a) != canonical_geometry_key(b)


def test_every_cache_key_param_is_actually_written():
    """ENABLE_ASPECT_RATIO_FILTERING was a phantom key hashing None forever.

    A cache-key param is "written" when some producer in ``src/`` actually
    sets that literal uppercase key on a params dict (``params["KEY"] = ...``
    or ``params.get("KEY", ...)`` read from an upstream writer). Bundled JSON
    presets store lowercase/renamed keys (e.g. ``subtraction_threshold`` ->
    ``THRESHOLD_VALUE``) that GUI/CLI config-mapping code re-emits uppercase,
    so the real check is "does any producer write this key", not "is this
    exact key present in the bundled default.json".
    """
    import pathlib

    from hydra_suite.core.inference.cache.keys import _BGSUB_KEY_PARAMS

    src_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "hydra_suite"
    corpus = ""
    for path in src_root.rglob("*.py"):
        if path.name == "keys.py" and path.parent.name == "cache":
            continue  # the reader itself doesn't count as a producer
        corpus += path.read_text(encoding="utf-8", errors="ignore")

    for key in _BGSUB_KEY_PARAMS:
        assert (
            f'"{key}"' in corpus or f"'{key}'" in corpus
        ), f"{key} is hashed into the cache key but nothing writes it"


# ---------------------------------------------------------------------------
# GPU/CPU Layer 2 fit parity (CNN/HeadTail on-device fit)
#
# The GPU (NVDEC on-device) branch of run_cnn_batch/run_headtail_batch must
# letterbox to the SAME geometry as the CPU branch -- same scale, same offset,
# same zero-padded canvas -- or a model trained on letterboxed crops sees an
# anisotropically-stretched crop on CUDA and a letterboxed one on CPU/MPS,
# a silent accuracy divergence invisible on this (non-CUDA) box. These tests
# exercise the shared torch seam's ``letterbox_fit`` on the CPU torch device
# (``F.interpolate`` runs there too) and compare shape + zero-padded band
# placement against the CPU ``apply_fit`` for the SAME ``FitResult`` -- the
# shape/offset arithmetic, not the resample kernel (grid_sample/interpolate
# != cv2 is an accepted, pre-existing identity-not-byte-identity gap
# elsewhere in this module).
# ---------------------------------------------------------------------------


def test_gpu_fit_matches_cpu_fit_shape_and_padding():
    import torch

    from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input
    from hydra_suite.core.canonicalization.resample import letterbox_fit

    # Non-square canonical canvas fit to a non-matching-aspect model input, so
    # BOTH scale and offset (letterbox padding) are exercised.
    canvas_wh = (60, 30)
    model_wh = (40, 40)
    fit = fit_to_model_input(canvas_wh, model_wh)
    assert fit.offset_xy != (0, 0), "sanity: this fit must actually pad"

    rng = np.random.default_rng(0)
    crop_hwc = rng.integers(1, 256, (canvas_wh[1], canvas_wh[0], 3)).astype(np.uint8)
    cpu_out = apply_fit(crop_hwc, fit)  # (model_h, model_w, 3) uint8

    batch = (
        torch.from_numpy(crop_hwc).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    )  # (1, 3, canvas_h, canvas_w)
    gpu_out = letterbox_fit(batch, fit.model_wh)  # (1, 3, model_h, model_w)

    model_h, model_w = fit.model_wh[1], fit.model_wh[0]
    assert gpu_out.shape == (1, 3, model_h, model_w)
    assert cpu_out.shape == (model_h, model_w, 3)

    ox, oy = fit.offset_xy
    inner_w, inner_h = fit.inner_wh
    pad_mask = np.ones((model_h, model_w), dtype=bool)
    pad_mask[oy : oy + inner_h, ox : ox + inner_w] = False

    # CPU letterbox pads with zeros outside the inner region.
    assert (cpu_out[pad_mask] == 0).all()
    # GPU letterbox pads with zeros at the SAME positions.
    gpu_np = gpu_out[0].permute(1, 2, 0).numpy()
    assert np.allclose(gpu_np[pad_mask], 0.0)


def test_run_cnn_batch_gpu_and_cpu_branches_share_one_fit(monkeypatch):
    """Both branches of run_cnn_batch must derive their target from the SAME
    fit_to_model_input(geometry.canvas_wh, model_wh) call -- not two
    independently-computed fits that could silently drift apart."""
    import torch

    from hydra_suite.core.canonicalization.fit import fit_to_model_input
    from hydra_suite.core.inference.stages import cnn as cnn_stage
    from hydra_suite.core.inference.stages import crops as crops_mod

    geometry = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    model = cnn_stage.CNNModel(
        backend=None,  # unused: predict_batch_cuda is stubbed below
        input_size=(40, 40),
        factor_names=["f"],
        factor_class_names=[["a", "b"]],
    )
    expected_model_wh = fit_to_model_input(geometry.canvas_wh, (40, 40)).model_wh

    class _FakeBatch:
        crops = torch.zeros((1, 3, geometry.canvas_h, geometry.canvas_w))
        obb_by_frame = {0: _obb(1, None)}

        def select_frame(self, f):
            return np.array([0])

    seen_model_whs = []
    import hydra_suite.core.canonicalization.resample as resample_mod

    real_letterbox_fit = resample_mod.letterbox_fit

    def _spy_letterbox_fit(crop_chw, model_wh):
        seen_model_whs.append(model_wh)
        return real_letterbox_fit(crop_chw, model_wh)

    monkeypatch.setattr(resample_mod, "letterbox_fit", _spy_letterbox_fit)
    monkeypatch.setattr(
        crops_mod, "extract_canonical_crops_batch", lambda *a, **k: _FakeBatch()
    )
    monkeypatch.setattr(crops_mod, "frames_on_cuda", lambda r, f: True)

    class _Backend:
        metadata = type("M", (), {"fit_policy": "letterbox"})()

        def predict_batch_cuda(self, crops, input_is_bgr=True):
            return [[np.array([0.5, 0.5], np.float32)]]

    model.backend = _Backend()
    cfg = cnn_stage.CNNConfig(label="x", model_path="/m.pt")
    rt = type("RT", (), {"device": "cpu"})()

    cnn_stage.run_cnn_batch([None], [_obb(1, None)], model, cfg, rt, geometry)

    assert len(seen_model_whs) == 1
    assert seen_model_whs[0] == expected_model_wh


# ---------------------------------------------------------------------------
# Non-square canonical canvas: extract_canonical_crops is device-agnostic and
# uses the SAME shape on a rectangular canvas as on a square one -- the crop
# builder's single torch seam (canonical_warp_batch) must never silently
# collapse to a square/otherwise-transposed shape.
# ---------------------------------------------------------------------------


def test_crops_nonsquare_canvas_uniform():
    from hydra_suite.core.inference.runtime import RuntimeContext
    from hydra_suite.core.inference.stages.crops import extract_canonical_crops

    geom = CanonicalGeometry.from_reference(60, 2.0, 1.3)  # non-square canvas
    frame = np.random.default_rng(0).integers(0, 256, (300, 300, 3), np.uint8)
    runtime = RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False)
    crops = extract_canonical_crops(frame, _obb(3, None), geom, runtime)
    assert crops.shape == (3, 3, geom.canvas_h, geom.canvas_w)
    assert geom.canvas_h != geom.canvas_w
