import numpy as np

from hydra_suite.core.identity.pose.vitpose.decode import decode_udp_cv2
from hydra_suite.core.identity.pose.vitpose.training.targets import (
    generate_udp_gaussian,
)

HM_WH = (48, 64)  # (W, H)


def test_encode_decode_roundtrip_subpixel():
    # subpixel centers, comfortably inside the map
    joints = np.array([[10.3, 20.7], [30.9, 40.1], [5.0, 5.0]], dtype=np.float32)
    vis = np.ones(3, dtype=np.float32)
    target, weight = generate_udp_gaussian(joints, vis, HM_WH, sigma=2.0)
    assert target.shape == (3, 64, 48)
    assert weight.shape == (3, 1)
    coords, maxvals = decode_udp_cv2(target[None, ...], kernel=11)  # (1,K,2)
    rec = coords[0]
    assert np.allclose(rec, joints, atol=0.25), f"{rec} vs {joints}"


def test_invisible_joint_zeroed():
    joints = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    vis = np.array([1.0, 0.0], dtype=np.float32)
    target, weight = generate_udp_gaussian(joints, vis, HM_WH, sigma=2.0)
    assert weight[1, 0] == 0.0
    assert target[1].max() == 0.0
    assert target[0].max() > 0.9  # peak ~1.0 at the visible joint
