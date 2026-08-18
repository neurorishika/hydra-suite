import numpy as np

from hydra_suite.core.post.processing import _fix_heading_globally


def _majority_orientation_matches_raw(theta):
    out = _fix_heading_globally(theta)
    # fraction of valid frames whose corrected heading stays near the raw heading
    d = np.abs(np.angle(np.exp(1j * (out - theta))))  # circular distance to raw
    return np.mean(d[~np.isnan(theta)] < (np.pi / 2))


def test_anchor_picks_raw_majority_orientation():
    # 20 frames pointing ~0.1 rad, 3 spurious 180-flips: majority head call is 0.1
    rng = np.random.default_rng(0)
    theta = np.full(23, 0.1) + rng.normal(0, 1e-3, 23)
    theta[5] = (0.1 + np.pi) % (2 * np.pi)
    theta[11] = (0.1 + np.pi) % (2 * np.pi)
    theta[17] = (0.1 + np.pi) % (2 * np.pi)
    # >= half the frames must end up agreeing with the raw majority (0.1), NOT flipped
    assert _majority_orientation_matches_raw(theta) >= 0.5


def test_anchor_is_stable_under_subpixel_jitter():
    # The global orientation must NOT flip when inputs move by ~1e-3 rad.
    rng = np.random.default_rng(1)
    base = np.full(40, 2.0)
    base[::7] = (2.0 + np.pi) % (2 * np.pi)  # a minority of flips
    outs = []
    for k in range(8):
        jitter = rng.normal(0, 1e-3, base.size)
        out = _fix_heading_globally(base + jitter)
        # record global orientation as the mean cos/sin (sign-stable summary)
        outs.append(np.mean(np.cos(out)))  # orientation-sign summary
    outs = np.array(outs)
    # all runs agree on orientation sign (no π-flip across jitter seeds)
    assert np.all(np.sign(outs) == np.sign(outs[0]))
