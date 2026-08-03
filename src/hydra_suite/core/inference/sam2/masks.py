"""Binary mask -> simplified largest external contour (SAM2 escalation)."""

from __future__ import annotations

import cv2
import numpy as np


def mask_to_contour(
    mask: np.ndarray,
    epsilon_frac: float = 0.01,
    min_points: int = 6,
    min_area: float = 4.0,
) -> np.ndarray | None:
    """Largest external contour of a binary mask as an (P, 2) float32 array.

    Simplified with approxPolyDP (epsilon = epsilon_frac * perimeter). Returns
    None when the mask is empty or the largest contour is degenerate. Single
    contour only (YOLO-seg has no holes).
    """
    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    eps = epsilon_frac * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
    if approx.shape[0] < min_points:
        approx = c.reshape(-1, 2).astype(np.float32)  # keep detail if oversimplified
    return approx
