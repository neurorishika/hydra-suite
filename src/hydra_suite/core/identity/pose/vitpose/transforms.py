"""Top-down pre/post-processing, UDP variant.

UDP (Unbiased Data Processing, Huang et al. CVPR 2020) defines unit length as
pixel SPACING (size - 1) rather than pixel count. Every released ViTPose
checkpoint sets use_udp=True, so warp, encode, and decode must all agree; mixing
costs ~1-2 AP silently. get_warp_matrix / transform_preds are transcribed from
upstream mmpose/core/post_processing/post_transforms.py.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .config import (
    IMAGE_SIZE_WH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PADDING_FACTOR,
    PIXEL_STD,
)


def box2cs(box_xywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, w, h = box_xywh[:4]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    aspect = IMAGE_SIZE_WH[0] / IMAGE_SIZE_WH[1]
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    scale = np.array([w, h], dtype=np.float32) / PIXEL_STD
    scale = scale * PADDING_FACTOR
    return center, scale


def get_warp_matrix(
    theta: float,
    size_input: np.ndarray,
    size_dst: np.ndarray,
    size_target: np.ndarray,
) -> np.ndarray:
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        np.ndarray: A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (
        -0.5 * size_input[0] * math.cos(theta)
        + 0.5 * size_input[1] * math.sin(theta)
        + 0.5 * size_target[0]
    )
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (
        -0.5 * size_input[0] * math.sin(theta)
        - 0.5 * size_input[1] * math.cos(theta)
        + 0.5 * size_target[1]
    )
    return matrix


def affine_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float = 0.0,
) -> np.ndarray:
    """The 2x3 UDP warp matrix top_down_affine applies. Exposed so tests can
    assert the UDP corner correspondence, which is invisible from the warped
    image alone."""
    w, h = IMAGE_SIZE_WH
    return get_warp_matrix(
        rot,
        center * 2.0,
        np.array([w, h], dtype=np.float32) - 1.0,  # UDP: unit = pixel spacing
        scale * PIXEL_STD,
    )


def top_down_affine(
    img: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    rot: float = 0.0,
) -> np.ndarray:
    w, h = IMAGE_SIZE_WH
    trans = affine_matrix(center, scale, rot)
    return cv2.warpAffine(img, trans, (w, h), flags=cv2.INTER_LINEAR)


def normalize(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN, np.float32)) / np.array(
        IMAGENET_STD, np.float32
    )
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def transform_preds(
    coords: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    output_size_wh: tuple[int, int],
) -> np.ndarray:
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image (UDP branch, use_udp=True).

    Transcribed verbatim from upstream
    mmpose/core/post_processing/post_transforms.py::transform_preds, taking
    only the use_udp=True branch. The UDP branch divides by
    (output_size - 1.0); the non-UDP branch does not. That single -1 is
    worth ~1 AP.

    Args:
        coords (np.ndarray[K, ndims]): Predicted keypoint locations.
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box wrt [width, height].
        output_size_wh (np.ndarray[2, ] | tuple(2,)): Size of the destination
            heatmaps.

    Returns:
        np.ndarray: Predicted coordinates in the image.
    """
    output_size = output_size_wh
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of PIXEL_STD (200).
    scale = scale * PIXEL_STD

    scale_x = scale[0] / (output_size[0] - 1.0)
    scale_y = scale[1] / (output_size[1] - 1.0)

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords
