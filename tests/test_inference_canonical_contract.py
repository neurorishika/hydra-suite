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
