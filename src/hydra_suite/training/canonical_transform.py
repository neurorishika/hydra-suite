"""The one transform training and inference share.

A torchvision Resize here and a cv2.resize there is exactly how train and
inference drift apart: PIL antialiases on downscale, cv2.INTER_LINEAR does not.
Both ends now call this transform, which wraps Layer 2
(``fit_to_model_input`` + ``apply_fit``) unmodified -- it introduces no
resampling or padding policy of its own.
"""

from __future__ import annotations

import numpy as np

from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input


class CanonicalFitTransform:
    """Fit a uint8 BGR image into ``model_hw`` (H, W) by isotropic letterbox.

    Callable, torchvision-compatible (``transform(image) -> np.ndarray``).
    Raises ``TypeError`` on non-uint8 input rather than silently converting --
    dtype coercion is exactly the kind of implicit behaviour that lets train
    and inference preprocessing drift apart unnoticed.
    """

    def __init__(self, model_hw: tuple[int, int]) -> None:
        self.model_hw = (int(model_hw[0]), int(model_hw[1]))

    def __call__(self, image) -> np.ndarray:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            raise TypeError(
                f"CanonicalFitTransform requires uint8 input, got {arr.dtype}"
            )
        h, w = arr.shape[:2]
        fit = fit_to_model_input((w, h), (self.model_hw[1], self.model_hw[0]))
        return apply_fit(arr, fit)


def cv2_bgr_loader(path) -> np.ndarray:
    """Decode an image file as uint8 BGR -- the format ``CanonicalFitTransform``
    (and the rest of the Layer 2 contract) expects.

    Unlike ``torchvision.datasets.folder.default_loader`` (PIL), ``cv2.imread``
    applies EXIF orientation automatically, so operator-supplied camera JPEGs
    decode the same way for training as they do at inference.
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def bgr_to_rgb_pil(arr: np.ndarray):
    """Convert a fitted BGR uint8 array to a PIL RGB image.

    Bridges ``CanonicalFitTransform``'s BGR output into the remaining
    PIL-based torchvision augmentations (flip/jitter/grayscale), which expect
    RGB order to match the ImageNet normalization stats.
    """
    import cv2
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
