"""Isolated 180-degree heading-flip correction.

Lives here, outside ``core``, so that both ``core.post.processing`` and
``core.individual.dataset.oriented_video`` can use it without forming an
import cycle: ``post.processing`` imports ``core.individual.identity``, which
executes ``core/individual/__init__`` -> ``dataset.generator`` ->
``dataset.oriented_video``, which used to import back into the
still-initialising ``post.processing``.
"""

from __future__ import annotations

import numpy as np


def _circ_diff(a, b):
    """Absolute circular difference in [0, pi]."""
    two_pi = 2.0 * np.pi
    d = abs(a - b) % two_pi
    return min(d, two_pi - d)


def _find_prev_valid(result, i):
    """Find the last non-NaN index before i, or -1."""
    prev_idx = i - 1
    while prev_idx >= 0 and np.isnan(result[prev_idx]):
        prev_idx -= 1
    return prev_idx


def _find_next_valid(result, start, n):
    """Find the first non-NaN index at or after start, or n."""
    idx = start
    while idx < n and np.isnan(result[idx]):
        idx += 1
    return idx


def _measure_flip_burst(result, start, n, prev_val, max_burst):
    """Find the end of a contiguous flipped burst starting at 'start'."""
    burst_end = start + 1
    while burst_end < n and burst_end - start < max_burst + 1:
        if np.isnan(result[burst_end]):
            burst_end += 1
            continue
        if _circ_diff(result[burst_end], prev_val) > np.pi / 2:
            burst_end += 1
        else:
            break
    return burst_end


def _fix_heading_flips(theta: np.ndarray, max_burst: int = 5) -> np.ndarray:
    """Detect and correct isolated 180-degree heading flips in a trajectory.

    Scans the theta array for contiguous bursts of frames where the heading
    jumped ~180° relative to the surrounding values and then returned.  Such
    bursts are corrected by adding π (mod 2π).

    Parameters
    ----------
    theta : np.ndarray
        Heading values in radians.  May contain NaN for missing frames.
    max_burst : int
        Maximum length (in frames) of a flip burst to correct.  Longer
        segments are assumed to be genuine orientation changes.

    Returns
    -------
    np.ndarray
        Corrected heading array (copy).
    """
    two_pi = 2.0 * np.pi
    result = theta.copy()
    n = len(result)
    if n < 3:
        return result

    # Iterative passes — a single pass may leave residual flips when bursts
    # are adjacent.  Three passes is sufficient for typical data.
    for _pass in range(3):
        changed = False
        i = 0
        while i < n:
            if np.isnan(result[i]):
                i += 1
                continue

            prev_idx = _find_prev_valid(result, i)
            if prev_idx < 0:
                i += 1
                continue

            prev_val = result[prev_idx]
            if _circ_diff(result[i], prev_val) <= np.pi / 2:
                i += 1
                continue

            # Found a potential flip burst starting at i
            burst_end = _measure_flip_burst(result, i, n, prev_val, max_burst)
            burst_len = burst_end - i

            if burst_len <= max_burst:
                post_idx = _find_next_valid(result, burst_end, n)
                if post_idx < n and _circ_diff(result[post_idx], prev_val) <= np.pi / 2:
                    for j in range(i, burst_end):
                        if not np.isnan(result[j]):
                            result[j] = (result[j] + np.pi) % two_pi
                    changed = True
                    i = burst_end
                    continue

            i = burst_end if burst_len > max_burst else i + 1

        if not changed:
            break

    return result
