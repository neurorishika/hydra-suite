"""256x256 ViTPose construction, strict loading, and mmpose-`default` decode.

The collaborator's checkpoints are ViT-base + TopdownHeatmapSimpleHead, which
is byte-for-byte our `ViT` + `ClassicHead` -- only the input resolution differs
from the repo's baked 192x256. `ViT` already takes `img_size_hw`, so the model
needs no repo change; only pre/post-processing is reimplemented here.

Decode is mmpose 0.x `post_process='default'`: argmax plus a +/-0.25 px
quarter-offset toward the brighter neighbour. NOT DARK, NOT UDP -- their config
sets `post_process='default'`, under which `modulate_kernel=11` is never read.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.core.multiarray as _np_multiarray
import torch

from hydra_suite.core.individual.pose.vitpose.config import VARIANTS
from hydra_suite.core.individual.pose.vitpose.heads import ClassicHead
from hydra_suite.core.individual.pose.vitpose.model import ViT
from hydra_suite.core.individual.pose.vitpose.vitpose import ViTPose

# mmpose checkpoints carry numpy scalars in `meta`, which weights_only=True
# rejects. Allowlist exactly those numpy primitives -- never weights_only=False,
# which would unpickle arbitrary code from a downloaded file. numpy 2.x resolves
# `numpy.core` to `numpy._core`, so the pickled NAME must be given explicitly.
# Both spellings are allowlisted: real (older-numpy) mmpose checkpoints pickle
# `numpy.core.multiarray.scalar`, while checkpoints written under this numpy
# (2.x, e.g. in tests) pickle `numpy._core.multiarray.scalar`.
#
# This allowlist's safety depends on numpy's own object-dtype hardening:
# `multiarray.scalar` together with `np.dtype` would be a remote-code-execution
# gadget on sufficiently old numpy, where `scalar` given an object dtype ran
# `pickle.loads` on its payload. numpy 2.x blocks that path, which is what makes
# allowlisting these primitives (rather than falling back to weights_only=False)
# safe here.
#
# The real collaborator checkpoints (older numpy) also pickle their scalar
# dtype as a concrete `numpy.dtypes.*DType` class (e.g. Float64DType) rather
# than only via `numpy.dtype` -- both forms are needed, confirmed empirically
# by loading the real ~1GB checkpoints (removing this loop breaks that load).
_SAFE_GLOBALS = [
    (_np_multiarray.scalar, "numpy.core.multiarray.scalar"),
    _np_multiarray.scalar,
    (np.dtype, "numpy.dtype"),
]
for _name in ("Float64DType", "Float32DType", "Int64DType", "Int32DType", "BoolDType"):
    _dtype = getattr(np.dtypes, _name, None)
    if _dtype is not None:
        _SAFE_GLOBALS.append(_dtype)

IMAGE_PX = 256
HEATMAP_PX = 64
VARIANT = "B"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_external_vitpose(num_keypoints: int) -> ViTPose:
    v = VARIANTS[VARIANT]
    backbone = ViT(
        embed_dim=v.embed_dim,
        depth=v.depth,
        num_heads=v.num_heads,
        img_size_hw=(IMAGE_PX, IMAGE_PX),
        drop_path_rate=v.drop_path_rate,
    )
    return ViTPose(backbone, ClassicHead(v.embed_dim, num_keypoints))


def infer_num_keypoints(state: dict) -> int:
    key = "keypoint_head.final_layer.weight"
    if key not in state:
        raise KeyError(f"checkpoint has no {key!r}; not a ViTPose heatmap head")
    return int(state[key].shape[0])


def load_external_checkpoint(path: Path) -> tuple[ViTPose, int]:
    """Strict load. A strict failure is a finding, not something to silence."""
    # Registered here (not at import time) so importing this module does not
    # mutate global torch.serialization state for the whole process; safe to
    # call repeatedly, add_safe_globals is idempotent.
    torch.serialization.add_safe_globals(_SAFE_GLOBALS)
    blob = torch.load(str(path), map_location="cpu", weights_only=True)
    state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    num_keypoints = infer_num_keypoints(state)
    model = build_external_vitpose(num_keypoints)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, num_keypoints


def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def decode_default(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mmpose 0.x `post_process='default'` decode, in crop pixels.

    Returns (coords[N, K, 2] in crop pixels, confidences[N, K]).
    """
    n, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(n, k, -1)
    idx = np.argmax(flat, axis=2)
    conf = np.take_along_axis(flat, idx[..., None], axis=2).squeeze(2)

    coords = np.zeros((n, k, 2), dtype=np.float32)
    coords[..., 0] = idx % w
    coords[..., 1] = idx // w

    for i in range(n):
        for j in range(k):
            px, py = int(coords[i, j, 0]), int(coords[i, j, 1])
            if 1 < px < w - 1 and 1 < py < h - 1:
                hm = heatmaps[i, j]
                dx = hm[py, px + 1] - hm[py, px - 1]
                dy = hm[py + 1, px] - hm[py - 1, px]
                coords[i, j, 0] += np.sign(dx) * 0.25
                coords[i, j, 1] += np.sign(dy) * 0.25

    coords *= IMAGE_PX / HEATMAP_PX
    return coords, conf


def predict(
    model: ViTPose, crops_bgr: list[np.ndarray], device: str
) -> tuple[np.ndarray, np.ndarray]:
    batch = np.stack([preprocess(c) for c in crops_bgr])
    tensor = torch.from_numpy(batch).to(device)
    model = model.to(device)
    with torch.no_grad():
        heatmaps = model(tensor).float().cpu().numpy()
    return decode_default(heatmaps)
