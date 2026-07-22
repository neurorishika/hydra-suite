import numpy as np
import torch

from hydra_suite.core.identity.pose.vitpose.decode import (
    decode_udp_cv2,
    decode_udp_torch,
)


def test_decode_torch_matches_cv2_oracle():
    # the two decoders must agree sub-pixel (the leaf's own gate, re-asserted)
    rng = np.random.default_rng(0)
    hm = rng.random((2, 4, 64, 48)).astype(np.float32)
    c_t, v_t = decode_udp_torch(torch.from_numpy(hm))
    c_c, v_c = decode_udp_cv2(hm)
    assert np.max(np.abs(c_t.numpy() - c_c)) < 0.1  # sub-pixel


def test_parity_harness_importable():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "verify_vitpose_runtimes",
        root / "tools" / "equivalence" / "verify_vitpose_runtimes.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "compare_runtimes")
