# External ViTPose Checkpoint Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone tool that runs a collaborator's external mmpose-trained ViTPose checkpoints (ant 9-keypoint, fly 29-keypoint) on individual animals cropped from our DEMO ant and fly videos, and renders skeleton-overlay contact sheets for qualitative judgement.

**Architecture:** A new `tools/vitpose/external_ckpt/` package reuses the repo's already-verified pure-torch `ViT` backbone and `ClassicHead` from `src/hydra_suite/core/individual/pose/vitpose/`, constructed at 256x256 instead of the repo default 192x256. It supplies its own top-down crop geometry (sourced from existing tracking CSVs, not a fresh detector run) and its own mmpose-`default` heatmap decode, because the repo's `transforms.py`/`decode.py` are baked to 192x256 + UDP. Nothing under `src/` is modified, so the byte-identical tracking equivalence guarantees are untouched.

**Tech Stack:** Python 3, PyTorch (MPS), OpenCV, NumPy, pandas, pytest. No mmpose / mmcv dependency — the collaborator's mmpose config files are converted once into plain JSON skeleton descriptors.

## Global Constraints

- **Do not modify anything under `src/`.** This tool is read-only with respect to production code. If a repo function cannot be reused as-is, reimplement it locally in `tools/vitpose/external_ckpt/`.
- Import repo code only from `hydra_suite.core.individual.pose.vitpose.{model,heads,vitpose,transforms,decode}`.
- Checkpoint architecture, fixed by the collaborator's configs and **not** configurable: ViT-base (`embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.3`), `patch_size=16`, input **256x256 (H, W)**, heatmap **64x64**, head = `ClassicHead` (`num_deconv_layers=2, filters=(256,256), kernels=(4,4), final_conv_kernel=1`).
- Decode is mmpose 0.x `post_process='default'` — argmax plus a `±0.25` px quarter-offset. It is **not** DARK and **not** UDP. `modulate_kernel=11` in their config is dead config (mmpose 0.x reads it only when `post_process='unbiased'`).
- Preprocessing normalisation is ImageNet: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`, on RGB in `[0, 1]`.
- Checkpoints must load with `strict=True`. A strict-load failure is a real finding to report, not something to work around with `strict=False`.
- Colours in the collaborator's configs are **RGB**; OpenCV draws **BGR**. Convert at the boundary, once, in the skeleton loader.
- Environment: `conda activate hydra-mps`; default inference device `mps`.
- Run `make format` before each commit. Files live in `tools/`, which formatters cover.

## Fixed Reference Data

Species presets, read from the existing DEMO tracking configs (`ant_config.json`, `melanogaster_config.json`) — hardcode these values, do not re-read the configs at runtime:

| species | video | tracking CSV | `reference_body_size` |
|---|---|---|---|
| `ant` | `/Users/neurorishika/Projects/Rockefeller/Ruta/Presentation/DEMO/DEMO 3/ant.mp4` | `/Users/neurorishika/Projects/Rockefeller/Ruta/Presentation/DEMO/DEMO 3/ant_tracking_final.csv` | `76.81` |
| `fly` | `/Users/neurorishika/Projects/Rockefeller/Ruta/Presentation/DEMO/DEMO 4/melanogaster.mp4` | `/Users/neurorishika/Projects/Rockefeller/Ruta/Presentation/DEMO/DEMO 4/melanogaster_tracking_final.csv` | `104.14` |

Tracking CSV columns used: `TrajectoryID`, `X`, `Y`, `Theta`, `FrameID`, `State`.

Checkpoint download URLs (Dropbox shares; `dl=0` must be rewritten to `dl=1`):

- ant: `https://www.dropbox.com/scl/fi/leluaj3nukpygpl8et6nl/ViTPose_base_ant9kp_256x256.pth?rlkey=c3vqtuhl6cgwzz6972s5ya3kb&dl=1`
- fly: `https://www.dropbox.com/scl/fi/y5d45ux1oetcnp6q5e1oh/ViTPose_base_fly29kp_ImgAug_256x256.pth?rlkey=9a8rheap55lmfp7g8b16dvbur&dl=1`

## File Structure

- `tools/vitpose/external_ckpt/__init__.py` — empty package marker.
- `tools/vitpose/external_ckpt/skeleton.py` — `SkeletonSpec` dataclass + JSON loader. Knows keypoint names, per-keypoint colours, edges, edge colours. No torch, no cv2 drawing.
- `tools/vitpose/external_ckpt/skeletons/ant_9kp.json`, `.../fly_29kp.json` — converted from the collaborator's mmpose dataset configs.
- `tools/vitpose/external_ckpt/crops.py` — tracking-CSV sampling and the 2x3 crop warp matrix. Pure geometry + pandas; no torch.
- `tools/vitpose/external_ckpt/model.py` — 256x256 model construction, strict checkpoint loading, keypoint-count inference, mmpose-`default` decode.
- `tools/vitpose/external_ckpt/render.py` — skeleton overlay on a crop, contact-sheet assembly, per-keypoint confidence table.
- `tools/vitpose/external_ckpt/cli.py` — argparse entry point wiring everything; `python -m tools.vitpose.external_ckpt.cli`.
- `tests/test_vitpose_external_ckpt.py` — unit tests for skeleton, crops, model, render. Imported as `tools.vitpose.external_ckpt.*` (`tests/conftest.py:11` already puts the repo root on `sys.path`).

---

### Task 1: Skeleton descriptors

Convert the collaborator's two mmpose dataset configs into plain JSON so nothing downstream needs mmpose installed.

**Files:**
- Create: `tools/vitpose/external_ckpt/__init__.py`
- Create: `tools/vitpose/external_ckpt/skeleton.py`
- Create: `tools/vitpose/external_ckpt/skeletons/ant_9kp.json` (generated)
- Create: `tools/vitpose/external_ckpt/skeletons/fly_29kp.json` (generated)
- Test: `tests/test_vitpose_external_ckpt.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SkeletonSpec` (frozen dataclass with fields `name: str`, `keypoint_names: list[str]`, `keypoint_colors_bgr: list[tuple[int, int, int]]`, `skeleton_edges: list[tuple[int, int]]`, `edge_colors_bgr: list[tuple[int, int, int]]`, and property `num_keypoints: int`); `load_skeleton(path: Path) -> SkeletonSpec`; `builtin_skeleton(species: str) -> SkeletonSpec` where `species` is `"ant"` or `"fly"`.

- [ ] **Step 1: Fetch the collaborator's config files**

The source repo `tywei08/ViTPose_checkpoints` is private; `gh` is already authenticated for it.

```bash
mkdir -p /tmp/vitpose_src
cd /tmp/vitpose_src
for f in configs/_base_/datasets/ant_9kp.py configs/_base_/datasets/fly_29kp.py; do
  mkdir -p "$(dirname "$f")"
  gh api "repos/tywei08/ViTPose_checkpoints/contents/$f" --jq '.content' | base64 -d > "$f"
done
wc -l configs/_base_/datasets/*.py
```

Expected: `41 .../ant_9kp.py`, `201 .../fly_29kp.py`.

- [ ] **Step 2: Generate the JSON descriptors**

Run this converter from the repo root. It `exec`s each mmpose config (they are plain Python assigning a `dataset_info` dict) and flattens it.

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
mkdir -p tools/vitpose/external_ckpt/skeletons
python3 - <<'EOF'
import json, pathlib

PAIRS = [
    ("/tmp/vitpose_src/configs/_base_/datasets/ant_9kp.py",
     "tools/vitpose/external_ckpt/skeletons/ant_9kp.json"),
    ("/tmp/vitpose_src/configs/_base_/datasets/fly_29kp.py",
     "tools/vitpose/external_ckpt/skeletons/fly_29kp.json"),
]

for src, out in PAIRS:
    ns = {}
    exec(pathlib.Path(src).read_text(), ns)
    info = ns["dataset_info"]
    kp = info["keypoint_info"]
    names = [kp[i]["name"] for i in sorted(kp)]
    colors = [kp[i]["color"] for i in sorted(kp)]
    sk = info["skeleton_info"]
    edges = [[names.index(sk[i]["link"][0]), names.index(sk[i]["link"][1])]
             for i in sorted(sk)]
    edge_colors = [sk[i]["color"] for i in sorted(sk)]
    payload = {
        "name": info["dataset_name"],
        "num_keypoints": len(names),
        "keypoint_names": names,
        "keypoint_colors_rgb": colors,
        "skeleton_edges": edges,
        "edge_colors_rgb": edge_colors,
    }
    pathlib.Path(out).write_text(json.dumps(payload, indent=2) + "\n")
    print(out, len(names), "kp,", len(edges), "edges")
EOF
```

Expected output:

```
tools/vitpose/external_ckpt/skeletons/ant_9kp.json 9 kp, 8 edges
tools/vitpose/external_ckpt/skeletons/fly_29kp.json 29 kp, 28 edges
```

Sanity-check the ant file: `keypoint_names` must be exactly
`["A_R_T", "A_L_T", "A_R_M", "A_L_M", "Head_T", "Centroid", "Abd_T", "Abd_B", "Head_B"]`
and `skeleton_edges` must be `[[0,2],[1,3],[2,4],[3,4],[4,8],[8,5],[5,6],[6,7]]`.
The fly file's first four names must be `["headTop", "thoraxCenter", "abdomenTop", "abdomenCenter"]`.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_vitpose_external_ckpt.py`:

```python
"""Unit tests for the external-ViTPose-checkpoint probe tool."""

from __future__ import annotations

import numpy as np
import pytest

from tools.vitpose.external_ckpt.skeleton import builtin_skeleton


def test_ant_skeleton_has_nine_named_keypoints():
    spec = builtin_skeleton("ant")
    assert spec.num_keypoints == 9
    assert spec.keypoint_names == [
        "A_R_T",
        "A_L_T",
        "A_R_M",
        "A_L_M",
        "Head_T",
        "Centroid",
        "Abd_T",
        "Abd_B",
        "Head_B",
    ]
    assert spec.skeleton_edges == [
        (0, 2), (1, 3), (2, 4), (3, 4), (4, 8), (8, 5), (5, 6), (6, 7)
    ]


def test_fly_skeleton_has_twentynine_keypoints_and_legs():
    spec = builtin_skeleton("fly")
    assert spec.num_keypoints == 29
    assert spec.keypoint_names[:4] == [
        "headTop",
        "thoraxCenter",
        "abdomenTop",
        "abdomenCenter",
    ]
    assert "hindlegRight" in spec.keypoint_names
    assert len(spec.skeleton_edges) == 28


def test_skeleton_colors_are_bgr_reversed_from_config_rgb():
    # ant keypoint 0 (A_R_T) is RGB [148, 0, 211] in the mmpose config.
    spec = builtin_skeleton("ant")
    assert spec.keypoint_colors_bgr[0] == (211, 0, 148)
    assert len(spec.keypoint_colors_bgr) == spec.num_keypoints
    assert len(spec.edge_colors_bgr) == len(spec.skeleton_edges)


def test_edges_index_within_range():
    for species in ("ant", "fly"):
        spec = builtin_skeleton(species)
        for a, b in spec.skeleton_edges:
            assert 0 <= a < spec.num_keypoints
            assert 0 <= b < spec.num_keypoints


def test_unknown_species_rejected():
    with pytest.raises(ValueError, match="unknown species"):
        builtin_skeleton("beetle")
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/test_vitpose_external_ckpt.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'tools.vitpose.external_ckpt'`.

- [ ] **Step 5: Implement the skeleton module**

Create `tools/vitpose/external_ckpt/__init__.py` as an empty file, then `tools/vitpose/external_ckpt/skeleton.py`:

```python
"""Skeleton descriptors for the collaborator's external ViTPose checkpoints.

Converted once from their mmpose `dataset_info` configs into plain JSON so this
tool needs no mmpose/mmcv install. Config colours are RGB; we store BGR because
everything downstream draws with OpenCV.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SKELETON_DIR = Path(__file__).parent / "skeletons"

_BUILTIN = {
    "ant": "ant_9kp.json",
    "fly": "fly_29kp.json",
}


@dataclass(frozen=True)
class SkeletonSpec:
    name: str
    keypoint_names: list[str]
    keypoint_colors_bgr: list[tuple[int, int, int]]
    skeleton_edges: list[tuple[int, int]]
    edge_colors_bgr: list[tuple[int, int, int]]

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoint_names)


def _to_bgr(rgb: list[int]) -> tuple[int, int, int]:
    r, g, b = rgb
    return (int(b), int(g), int(r))


def load_skeleton(path: Path) -> SkeletonSpec:
    payload = json.loads(Path(path).read_text())
    names = list(payload["keypoint_names"])
    if payload["num_keypoints"] != len(names):
        raise ValueError(
            f"{path}: num_keypoints={payload['num_keypoints']} but "
            f"{len(names)} names"
        )
    return SkeletonSpec(
        name=payload["name"],
        keypoint_names=names,
        keypoint_colors_bgr=[_to_bgr(c) for c in payload["keypoint_colors_rgb"]],
        skeleton_edges=[(int(a), int(b)) for a, b in payload["skeleton_edges"]],
        edge_colors_bgr=[_to_bgr(c) for c in payload["edge_colors_rgb"]],
    )


def builtin_skeleton(species: str) -> SkeletonSpec:
    if species not in _BUILTIN:
        raise ValueError(
            f"unknown species {species!r} (expected one of {sorted(_BUILTIN)})"
        )
    return load_skeleton(SKELETON_DIR / _BUILTIN[species])
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_vitpose_external_ckpt.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
make format
git add tools/vitpose/external_ckpt tests/test_vitpose_external_ckpt.py
git commit -m "feat(tools): skeleton descriptors for external ViTPose checkpoints"
```

---

### Task 2: Crop sampling and warp geometry

Pick which animals to look at, and build the single warp matrix that takes a full frame straight to a 256x256 top-down crop. One `warpAffine` composes translate, rotate and scale so there is only one resample.

**Files:**
- Create: `tools/vitpose/external_ckpt/crops.py`
- Modify: `tests/test_vitpose_external_ckpt.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `CropSample` (frozen dataclass, fields `frame_id: int`, `track_id: int`, `cx: float`, `cy: float`, `theta: float`); `select_samples(csv_path: Path, n: int) -> list[CropSample]`; `crop_matrix(cx: float, cy: float, theta: float, side_px: float, out_px: int, rotate: bool) -> np.ndarray` returning a `(2, 3)` float32 matrix; `warp_crop(frame_bgr: np.ndarray, matrix: np.ndarray, out_px: int) -> np.ndarray`.

**Heading convention.** `Theta` in the tracking CSV is the heading in image coordinates (x right, y **down**), so the forward unit vector is `(cos θ, sin θ)`. "Upright" means forward points toward the top of the crop, i.e. direction `(0, -1)`. The point rotation needed is therefore by `δ = -π/2 - θ`. `cv2.getRotationMatrix2D(center, angle_deg, scale)` builds a matrix that rotates *points* by `-angle_deg` in this y-down frame, so `angle_deg = degrees(θ) + 90`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vitpose_external_ckpt.py`:

```python
import math

from tools.vitpose.external_ckpt.crops import (
    CropSample,
    crop_matrix,
    select_samples,
    warp_crop,
)


def _apply(matrix, x, y):
    v = np.array([x, y, 1.0], dtype=np.float64)
    return (float(matrix[0] @ v), float(matrix[1] @ v))


def test_crop_matrix_maps_center_to_output_center():
    for rotate in (False, True):
        m = crop_matrix(
            cx=100.0, cy=50.0, theta=1.234, side_px=80.0, out_px=256,
            rotate=rotate,
        )
        out = _apply(m, 100.0, 50.0)
        assert out == pytest.approx((128.0, 128.0), abs=1e-4)


def test_axis_mode_is_pure_scale_and_translate():
    m = crop_matrix(
        cx=100.0, cy=50.0, theta=2.0, side_px=80.0, out_px=256, rotate=False,
    )
    # Right edge of the source square maps to the right edge of the crop.
    assert _apply(m, 140.0, 50.0) == pytest.approx((256.0, 128.0), abs=1e-4)
    # Bottom edge maps to the bottom edge -- no rotation regardless of theta.
    assert _apply(m, 100.0, 90.0) == pytest.approx((128.0, 256.0), abs=1e-4)


@pytest.mark.parametrize("theta", [0.0, 1.0, 2.5, -0.7, math.pi])
def test_rotate_mode_puts_heading_at_top_of_crop(theta):
    cx, cy, side = 100.0, 50.0, 80.0
    m = crop_matrix(cx, cy, theta, side_px=side, out_px=256, rotate=True)
    # A point half a crop-width ahead along the heading...
    ahead_x = cx + (side / 2.0) * math.cos(theta)
    ahead_y = cy + (side / 2.0) * math.sin(theta)
    # ...must land at top-center of the crop.
    assert _apply(m, ahead_x, ahead_y) == pytest.approx((128.0, 0.0), abs=1e-3)


def test_warp_crop_returns_square_output_of_requested_size():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    frame[40:60, 90:110] = 255
    m = crop_matrix(100.0, 50.0, 0.0, side_px=80.0, out_px=256, rotate=False)
    crop = warp_crop(frame, m, 256)
    assert crop.shape == (256, 256, 3)
    assert crop.dtype == np.uint8
    assert crop[128, 128].tolist() == [255, 255, 255]


def _write_csv(tmp_path, rows):
    header = "TrajectoryID,X,Y,Theta,FrameID,State\n"
    body = "".join(
        f"{t},{x},{y},{th},{f},{s}\n" for t, x, y, th, f, s in rows
    )
    p = tmp_path / "track.csv"
    p.write_text(header + body)
    return p


def test_select_samples_returns_requested_count_and_spreads_over_frames(tmp_path):
    rows = []
    for frame in range(0, 100):
        for track in range(3):
            rows.append((track, 10 * track, 20 + frame, 0.5, frame, "active"))
    csv = _write_csv(tmp_path, rows)
    samples = select_samples(csv, n=12)
    assert len(samples) == 12
    frames = [s.frame_id for s in samples]
    assert frames == sorted(frames)
    assert len(set(frames)) == 12
    # Spread across the whole range, not clustered at the start.
    assert min(frames) < 10 and max(frames) > 89


def test_select_samples_varies_track_ids(tmp_path):
    rows = []
    for frame in range(0, 60):
        for track in range(4):
            rows.append((track, 10 * track, 20, 0.1, frame, "active"))
    csv = _write_csv(tmp_path, rows)
    samples = select_samples(csv, n=8)
    assert len(set(s.track_id for s in samples)) > 1


def test_select_samples_ignores_non_active_rows(tmp_path):
    rows = [(0, 1, 2, 0.0, f, "tentative") for f in range(50)]
    rows += [(1, 3, 4, 0.0, f, "active") for f in range(50)]
    csv = _write_csv(tmp_path, rows)
    samples = select_samples(csv, n=5)
    assert all(s.track_id == 1 for s in samples)


def test_select_samples_is_deterministic(tmp_path):
    rows = []
    for frame in range(0, 80):
        for track in range(3):
            rows.append((track, 5 * track, 7, 0.3, frame, "active"))
    csv = _write_csv(tmp_path, rows)
    assert select_samples(csv, n=9) == select_samples(csv, n=9)


def test_select_samples_raises_when_no_active_rows(tmp_path):
    csv = _write_csv(tmp_path, [(0, 1, 2, 0.0, 1, "lost")])
    with pytest.raises(ValueError, match="no active"):
        select_samples(csv, n=4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vitpose_external_ckpt.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'tools.vitpose.external_ckpt.crops'`.

- [ ] **Step 3: Implement the crops module**

Create `tools/vitpose/external_ckpt/crops.py`:

```python
"""Top-down crop sampling from existing tracking output.

We take crop centres and headings straight from a completed
`*_tracking_final.csv` rather than re-running a detector, so this probe costs
nothing but video seeks.

Heading convention: `Theta` is measured in image coordinates (x right, y DOWN),
so forward is `(cos t, sin t)`. "Upright" means forward points to `(0, -1)`,
which is a point rotation by `-pi/2 - t`. `cv2.getRotationMatrix2D` rotates
points by `-angle_deg` in this y-down frame, hence `angle_deg = deg(t) + 90`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CropSample:
    frame_id: int
    track_id: int
    cx: float
    cy: float
    theta: float


def select_samples(csv_path: Path, n: int) -> list[CropSample]:
    """Pick `n` (frame, track) pairs spread evenly over the tracked range.

    Deterministic: frames are taken at evenly spaced positions through the
    sorted unique active frames, and the k-th sample takes the k-th distinct
    track present in its frame (wrapping), so track IDs vary without any RNG.
    """
    df = pd.read_csv(
        csv_path,
        usecols=["TrajectoryID", "X", "Y", "Theta", "FrameID", "State"],
    )
    df = df[df["State"] == "active"]
    if df.empty:
        raise ValueError(f"{csv_path}: no active rows to sample")

    frames = np.sort(df["FrameID"].unique())
    if len(frames) < n:
        picks = frames
    else:
        idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
        picks = frames[np.unique(idx)]

    samples: list[CropSample] = []
    for k, frame_id in enumerate(picks):
        rows = df[df["FrameID"] == frame_id].sort_values("TrajectoryID")
        row = rows.iloc[k % len(rows)]
        samples.append(
            CropSample(
                frame_id=int(frame_id),
                track_id=int(row["TrajectoryID"]),
                cx=float(row["X"]),
                cy=float(row["Y"]),
                theta=float(row["Theta"]),
            )
        )
    return samples


def crop_matrix(
    cx: float,
    cy: float,
    theta: float,
    side_px: float,
    out_px: int,
    rotate: bool,
) -> np.ndarray:
    """2x3 affine taking the source square of `side_px` centred on (cx, cy) to
    an `out_px` x `out_px` crop, optionally rotating the heading to point up."""
    angle_deg = math.degrees(theta) + 90.0 if rotate else 0.0
    scale = out_px / side_px
    m = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    # getRotationMatrix2D pins (cx, cy); shift it to the crop centre.
    m[0, 2] += out_px / 2.0 - cx
    m[1, 2] += out_px / 2.0 - cy
    return m.astype(np.float32)


def warp_crop(
    frame_bgr: np.ndarray, matrix: np.ndarray, out_px: int
) -> np.ndarray:
    return cv2.warpAffine(
        frame_bgr, matrix, (out_px, out_px), flags=cv2.INTER_LINEAR
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vitpose_external_ckpt.py -v`
Expected: all pass — the 5 from Task 1 plus the new ones (the parametrised heading test contributes 5 cases).

- [ ] **Step 5: Commit**

```bash
make format
git add tools/vitpose/external_ckpt/crops.py tests/test_vitpose_external_ckpt.py
git commit -m "feat(tools): tracking-CSV crop sampling and 256px warp geometry"
```

---

### Task 3: Model construction, strict loading, and decode

**Files:**
- Create: `tools/vitpose/external_ckpt/model.py`
- Modify: `tests/test_vitpose_external_ckpt.py` (append)

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces: `IMAGE_PX = 256`; `HEATMAP_PX = 64`; `build_external_vitpose(num_keypoints: int) -> ViTPose`; `infer_num_keypoints(state: dict) -> int`; `load_external_checkpoint(path: Path) -> tuple[ViTPose, int]`; `preprocess(crop_bgr: np.ndarray) -> np.ndarray` returning `(3, 256, 256)` float32; `decode_default(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]` returning `(N, K, 2)` crop-pixel coords and `(N, K)` confidences; `predict(model, crops_bgr: list[np.ndarray], device: str) -> tuple[np.ndarray, np.ndarray]`.

Note: the repo's `ViT.__init__` takes `img_size_hw` as **(H, W)**, defaulting to `(256, 192)`. We pass `(256, 256)`. With `PATCH_SIZE=16` that yields a 16x16 patch grid and a `pos_embed` of shape `(1, 257, 768)` — exactly what the collaborator's checkpoint carries.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vitpose_external_ckpt.py`:

```python
import torch

from tools.vitpose.external_ckpt.model import (
    HEATMAP_PX,
    IMAGE_PX,
    build_external_vitpose,
    decode_default,
    infer_num_keypoints,
    load_external_checkpoint,
    preprocess,
)


def test_model_is_built_for_256_square_input():
    model = build_external_vitpose(9)
    assert model.backbone.pos_embed.shape == (1, 257, 768)
    assert model.keypoint_head.final_layer.weight.shape == (9, 256, 1, 1)


def test_model_forward_produces_64x64_heatmaps():
    model = build_external_vitpose(9).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, IMAGE_PX, IMAGE_PX))
    assert out.shape == (2, 9, HEATMAP_PX, HEATMAP_PX)


def test_infer_num_keypoints_reads_final_layer():
    model = build_external_vitpose(29)
    assert infer_num_keypoints(model.state_dict()) == 29


def test_load_external_checkpoint_round_trips_strictly(tmp_path):
    model = build_external_vitpose(9)
    ckpt = tmp_path / "fake.pth"
    torch.save({"state_dict": model.state_dict()}, ckpt)
    loaded, k = load_external_checkpoint(ckpt)
    assert k == 9
    for a, b in zip(
        model.state_dict().values(), loaded.state_dict().values()
    ):
        assert torch.equal(a, b)


def test_load_external_checkpoint_rejects_unexpected_keys(tmp_path):
    model = build_external_vitpose(9)
    state = dict(model.state_dict())
    state["backbone.bogus_key"] = torch.zeros(1)
    ckpt = tmp_path / "bad.pth"
    torch.save(state, ckpt)
    with pytest.raises(RuntimeError, match="bogus_key"):
        load_external_checkpoint(ckpt)


def test_preprocess_normalizes_rgb_with_imagenet_stats():
    crop = np.zeros((IMAGE_PX, IMAGE_PX, 3), dtype=np.uint8)
    crop[:, :, 2] = 255  # pure red in BGR
    out = preprocess(crop)
    assert out.shape == (3, IMAGE_PX, IMAGE_PX)
    assert out.dtype == np.float32
    # R channel: (1.0 - 0.485) / 0.229
    assert out[0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-4)
    # G channel: (0.0 - 0.456) / 0.224
    assert out[1, 0, 0] == pytest.approx((0.0 - 0.456) / 0.224, abs=1e-4)


def test_decode_default_recovers_peak_scaled_to_crop_pixels():
    hm = np.zeros((1, 1, HEATMAP_PX, HEATMAP_PX), dtype=np.float32)
    hm[0, 0, 20, 10] = 5.0  # (row=20, col=10)
    coords, conf = decode_default(hm)
    stride = IMAGE_PX / HEATMAP_PX
    assert coords.shape == (1, 1, 2)
    assert coords[0, 0] == pytest.approx([10 * stride, 20 * stride], abs=1e-4)
    assert conf[0, 0] == pytest.approx(5.0)


def test_decode_default_applies_quarter_pixel_offset_toward_brighter_neighbor():
    hm = np.zeros((1, 1, HEATMAP_PX, HEATMAP_PX), dtype=np.float32)
    hm[0, 0, 20, 10] = 5.0
    hm[0, 0, 20, 11] = 3.0  # brighter to the right than to the left
    hm[0, 0, 21, 10] = 2.0  # brighter below than above
    coords, _ = decode_default(hm)
    stride = IMAGE_PX / HEATMAP_PX
    assert coords[0, 0] == pytest.approx(
        [(10 + 0.25) * stride, (20 + 0.25) * stride], abs=1e-4
    )


def test_decode_default_skips_offset_at_border():
    hm = np.zeros((1, 1, HEATMAP_PX, HEATMAP_PX), dtype=np.float32)
    hm[0, 0, 0, 0] = 5.0
    coords, _ = decode_default(hm)
    assert coords[0, 0] == pytest.approx([0.0, 0.0], abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vitpose_external_ckpt.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'tools.vitpose.external_ckpt.model'`.

- [ ] **Step 3: Implement the model module**

Create `tools/vitpose/external_ckpt/model.py`:

```python
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
import torch

from hydra_suite.core.individual.pose.vitpose.config import VARIANTS
from hydra_suite.core.individual.pose.vitpose.heads import ClassicHead
from hydra_suite.core.individual.pose.vitpose.model import ViT
from hydra_suite.core.individual.pose.vitpose.vitpose import ViTPose

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vitpose_external_ckpt.py -v`
Expected: all pass, including the 9 new tests.

Note on `test_decode_default_skips_offset_at_border`: the guard is `1 < px`, not `0 < px`, matching upstream mmpose exactly — a peak at column 1 also gets no offset. That is upstream's behaviour and is deliberate.

- [ ] **Step 5: Commit**

```bash
make format
git add tools/vitpose/external_ckpt/model.py tests/test_vitpose_external_ckpt.py
git commit -m "feat(tools): 256px ViTPose build, strict load, mmpose-default decode"
```

---

### Task 4: Rendering

**Files:**
- Create: `tools/vitpose/external_ckpt/render.py`
- Modify: `tests/test_vitpose_external_ckpt.py` (append)

**Interfaces:**
- Consumes: `SkeletonSpec` from Task 1.
- Produces: `draw_pose(crop_bgr: np.ndarray, coords: np.ndarray, conf: np.ndarray, spec: SkeletonSpec, conf_thr: float = 0.2) -> np.ndarray`; `label_tile(tile_bgr: np.ndarray, text: str) -> np.ndarray`; `contact_sheet(tiles: list[np.ndarray], cols: int = 4, pad: int = 8) -> np.ndarray`; `confidence_table(conf: np.ndarray, spec: SkeletonSpec) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vitpose_external_ckpt.py`:

```python
from tools.vitpose.external_ckpt.render import (
    confidence_table,
    contact_sheet,
    draw_pose,
    label_tile,
)


def test_draw_pose_does_not_mutate_input_and_keeps_shape():
    spec = builtin_skeleton("ant")
    crop = np.zeros((256, 256, 3), dtype=np.uint8)
    original = crop.copy()
    coords = np.full((spec.num_keypoints, 2), 128.0, dtype=np.float32)
    conf = np.ones(spec.num_keypoints, dtype=np.float32)
    out = draw_pose(crop, coords, conf, spec)
    assert out.shape == crop.shape
    assert np.array_equal(crop, original)
    assert out.any()


def test_draw_pose_marks_pixels_near_a_confident_keypoint():
    spec = builtin_skeleton("ant")
    crop = np.zeros((256, 256, 3), dtype=np.uint8)
    coords = np.zeros((spec.num_keypoints, 2), dtype=np.float32)
    coords[:] = 200.0
    coords[0] = (40.0, 60.0)
    conf = np.zeros(spec.num_keypoints, dtype=np.float32)
    conf[0] = 0.9
    out = draw_pose(crop, coords, conf, spec)
    assert out[55:66, 35:46].any()


def test_draw_pose_skips_low_confidence_keypoints():
    spec = builtin_skeleton("ant")
    crop = np.zeros((256, 256, 3), dtype=np.uint8)
    coords = np.full((spec.num_keypoints, 2), 128.0, dtype=np.float32)
    conf = np.zeros(spec.num_keypoints, dtype=np.float32)
    out = draw_pose(crop, coords, conf, spec, conf_thr=0.2)
    assert not out.any()


def test_contact_sheet_grid_dimensions():
    tiles = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(12)]
    sheet = contact_sheet(tiles, cols=4, pad=8)
    # 4 cols, 3 rows, 8px padding on every side and between tiles.
    assert sheet.shape == (3 * 256 + 4 * 8, 4 * 256 + 5 * 8, 3)


def test_contact_sheet_pads_ragged_last_row():
    tiles = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(10)]
    sheet = contact_sheet(tiles, cols=4, pad=8)
    assert sheet.shape == (3 * 256 + 4 * 8, 4 * 256 + 5 * 8, 3)


def test_label_tile_adds_a_banner_and_preserves_width():
    tile = np.zeros((256, 256, 3), dtype=np.uint8)
    out = label_tile(tile, "f=100 t=3")
    assert out.shape[1] == 256
    assert out.shape[0] > 256
    assert out[:20].any()


def test_confidence_table_lists_every_keypoint_with_median():
    spec = builtin_skeleton("ant")
    conf = np.tile(
        np.linspace(0.1, 0.9, spec.num_keypoints, dtype=np.float32), (5, 1)
    )
    table = confidence_table(conf, spec)
    for name in spec.keypoint_names:
        assert name in table
    assert "median" in table.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vitpose_external_ckpt.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'tools.vitpose.external_ckpt.render'`.

- [ ] **Step 3: Implement the render module**

Create `tools/vitpose/external_ckpt/render.py`:

```python
"""Skeleton overlays and contact sheets for eyeballing checkpoint output."""

from __future__ import annotations

import cv2
import numpy as np

from .skeleton import SkeletonSpec

FONT = cv2.FONT_HERSHEY_SIMPLEX
BANNER_H = 24


def draw_pose(
    crop_bgr: np.ndarray,
    coords: np.ndarray,
    conf: np.ndarray,
    spec: SkeletonSpec,
    conf_thr: float = 0.2,
) -> np.ndarray:
    """Overlay edges and keypoints. Marker radius scales with confidence so a
    hesitant keypoint reads as small rather than as a confident mistake."""
    out = crop_bgr.copy()
    ok = conf >= conf_thr

    for (a, b), color in zip(spec.skeleton_edges, spec.edge_colors_bgr):
        if not (ok[a] and ok[b]):
            continue
        pa = (int(round(coords[a, 0])), int(round(coords[a, 1])))
        pb = (int(round(coords[b, 0])), int(round(coords[b, 1])))
        cv2.line(out, pa, pb, color, 1, cv2.LINE_AA)

    for i, color in enumerate(spec.keypoint_colors_bgr):
        if not ok[i]:
            continue
        radius = 2 + int(round(3 * min(float(conf[i]), 1.0)))
        center = (int(round(coords[i, 0])), int(round(coords[i, 1])))
        cv2.circle(out, center, radius, color, -1, cv2.LINE_AA)

    return out


def label_tile(tile_bgr: np.ndarray, text: str) -> np.ndarray:
    h, w = tile_bgr.shape[:2]
    banner = np.zeros((BANNER_H, w, 3), dtype=np.uint8)
    cv2.putText(
        banner, text, (4, 17), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )
    return np.vstack([banner, tile_bgr])


def contact_sheet(
    tiles: list[np.ndarray], cols: int = 4, pad: int = 8
) -> np.ndarray:
    if not tiles:
        raise ValueError("contact_sheet needs at least one tile")
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros(
        (rows * th + (rows + 1) * pad, cols * tw + (cols + 1) * pad, 3),
        dtype=np.uint8,
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = pad + r * (th + pad)
        x = pad + c * (tw + pad)
        sheet[y : y + th, x : x + tw] = tile
    return sheet


def confidence_table(conf: np.ndarray, spec: SkeletonSpec) -> str:
    """conf: (N, K) peak heatmap values. A keypoint the model is guessing at
    shows up immediately as a low median row."""
    lines = [f"{'keypoint':<22} {'median':>8} {'min':>8} {'max':>8}"]
    med = np.median(conf, axis=0)
    lo = conf.min(axis=0)
    hi = conf.max(axis=0)
    for i, name in enumerate(spec.keypoint_names):
        lines.append(
            f"{name:<22} {med[i]:>8.3f} {lo[i]:>8.3f} {hi[i]:>8.3f}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vitpose_external_ckpt.py -v`
Expected: all pass, including the 7 new tests.

- [ ] **Step 5: Commit**

```bash
make format
git add tools/vitpose/external_ckpt/render.py tests/test_vitpose_external_ckpt.py
git commit -m "feat(tools): skeleton overlay and contact-sheet rendering"
```

---

### Task 5: CLI and first real run

**Files:**
- Create: `tools/vitpose/external_ckpt/cli.py`
- Modify: `tests/test_vitpose_external_ckpt.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: `SPECIES: dict[str, SpeciesPreset]`; `read_frames(video_path: Path, frame_ids: list[int]) -> dict[int, np.ndarray]`; `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vitpose_external_ckpt.py`:

```python
from tools.vitpose.external_ckpt.cli import SPECIES, build_parser, read_frames


def test_species_presets_carry_video_csv_and_body_size():
    assert set(SPECIES) == {"ant", "fly"}
    assert SPECIES["ant"].body_size_px == pytest.approx(76.81)
    assert SPECIES["fly"].body_size_px == pytest.approx(104.14)
    assert SPECIES["ant"].video.name == "ant.mp4"
    assert SPECIES["fly"].video.name == "melanogaster.mp4"
    assert SPECIES["ant"].csv.name == "ant_tracking_final.csv"
    assert SPECIES["fly"].csv.name == "melanogaster_tracking_final.csv"


def test_parser_defaults_match_the_agreed_probe_shape():
    args = build_parser().parse_args(["--species", "ant"])
    assert args.n == 12
    assert args.scale == pytest.approx(2.0)
    assert args.device == "mps"
    assert args.out_px == 256


def test_read_frames_returns_requested_frames_in_order(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64)
    )
    for i in range(20):
        frame = np.full((64, 64, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frames = read_frames(path, [2, 9, 15])
    assert sorted(frames) == [2, 9, 15]
    for fid, frame in frames.items():
        assert frame.shape == (64, 64, 3)


def test_read_frames_raises_on_unreadable_video(tmp_path):
    bad = tmp_path / "nope.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="cannot open"):
        read_frames(bad, [0])
```

Add `import cv2` to the test file's imports if it is not already there.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vitpose_external_ckpt.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'tools.vitpose.external_ckpt.cli'`.

- [ ] **Step 3: Implement the CLI**

Create `tools/vitpose/external_ckpt/cli.py`:

```python
"""Run an external ViTPose checkpoint over sampled crops and write sheets.

    python -m tools.vitpose.external_ckpt.cli \
        --species ant --ckpt /path/ViTPose_base_ant9kp_256x256.pth \
        --out /tmp/vitpose_probe
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .crops import crop_matrix, select_samples, warp_crop
from .model import load_external_checkpoint, predict
from .render import confidence_table, contact_sheet, draw_pose, label_tile
from .skeleton import builtin_skeleton

DEMO = Path("/Users/neurorishika/Projects/Rockefeller/Ruta/Presentation/DEMO")


@dataclass(frozen=True)
class SpeciesPreset:
    video: Path
    csv: Path
    body_size_px: float


SPECIES: dict[str, SpeciesPreset] = {
    "ant": SpeciesPreset(
        video=DEMO / "DEMO 3" / "ant.mp4",
        csv=DEMO / "DEMO 3" / "ant_tracking_final.csv",
        body_size_px=76.81,
    ),
    "fly": SpeciesPreset(
        video=DEMO / "DEMO 4" / "melanogaster.mp4",
        csv=DEMO / "DEMO 4" / "melanogaster_tracking_final.csv",
        body_size_px=104.14,
    ),
}


def read_frames(video_path: Path, frame_ids: list[int]) -> dict[int, np.ndarray]:
    """Seek-and-grab. Frames are requested in ascending order so the decoder
    only ever moves forward."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    out: dict[int, np.ndarray] = {}
    try:
        for fid in sorted(set(frame_ids)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"cannot read frame {fid} of {video_path}")
            out[fid] = frame
    finally:
        cap.release()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--species", required=True, choices=sorted(SPECIES))
    p.add_argument("--ckpt", type=Path, help="external .pth checkpoint")
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--body-size", type=float, default=None, dest="body_size")
    p.add_argument("--n", type=int, default=12, help="samples per crop mode")
    p.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="crop side as a multiple of reference body size",
    )
    p.add_argument("--out-px", type=int, default=256, dest="out_px")
    p.add_argument("--device", default="mps")
    p.add_argument("--conf-thr", type=float, default=0.2, dest="conf_thr")
    p.add_argument("--out", type=Path, default=Path("/tmp/vitpose_probe"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preset = SPECIES[args.species]
    video = args.video or preset.video
    csv = args.csv or preset.csv
    body_size = args.body_size or preset.body_size_px
    side_px = args.scale * body_size

    spec = builtin_skeleton(args.species)
    model, num_keypoints = load_external_checkpoint(args.ckpt)
    if num_keypoints != spec.num_keypoints:
        raise SystemExit(
            f"checkpoint has {num_keypoints} keypoints but the {args.species} "
            f"skeleton declares {spec.num_keypoints}"
        )
    print(f"loaded {args.ckpt.name}: {num_keypoints} keypoints, strict OK")

    samples = select_samples(csv, args.n)
    frames = read_frames(video, [s.frame_id for s in samples])
    args.out.mkdir(parents=True, exist_ok=True)

    for mode, rotate in (("axis", False), ("rot", True)):
        crops = [
            warp_crop(
                frames[s.frame_id],
                crop_matrix(
                    s.cx, s.cy, s.theta, side_px, args.out_px, rotate
                ),
                args.out_px,
            )
            for s in samples
        ]
        coords, conf = predict(model, crops, args.device)
        tiles = [
            label_tile(
                draw_pose(
                    crops[i], coords[i], conf[i], spec, conf_thr=args.conf_thr
                ),
                f"f={samples[i].frame_id} t={samples[i].track_id}",
            )
            for i in range(len(crops))
        ]
        sheet_path = args.out / f"{args.species}_{mode}.png"
        cv2.imwrite(str(sheet_path), contact_sheet(tiles, cols=4))
        table_path = args.out / f"{args.species}_{mode}_confidence.txt"
        table_path.write_text(confidence_table(conf, spec) + "\n")
        print(f"wrote {sheet_path}")
        print(f"wrote {table_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vitpose_external_ckpt.py -v`
Expected: all pass.

- [ ] **Step 5: Download the checkpoints**

```bash
mkdir -p /tmp/vitpose_external
cd /tmp/vitpose_external
curl -L -o ViTPose_base_ant9kp_256x256.pth \
  "https://www.dropbox.com/scl/fi/leluaj3nukpygpl8et6nl/ViTPose_base_ant9kp_256x256.pth?rlkey=c3vqtuhl6cgwzz6972s5ya3kb&dl=1"
curl -L -o ViTPose_base_fly29kp_ImgAug_256x256.pth \
  "https://www.dropbox.com/scl/fi/y5d45ux1oetcnp6q5e1oh/ViTPose_base_fly29kp_ImgAug_256x256.pth?rlkey=9a8rheap55lmfp7g8b16dvbur&dl=1"
ls -lh *.pth
shasum -a 256 *.pth
```

Expected: two files of roughly 300-700 MB. A file of a few KB means Dropbox returned an HTML interstitial instead of the binary — check with `file *.pth`; if so, stop and ask the user to download them manually rather than guessing at another URL form.

Record the two SHA-256 values in the run notes (Step 8) so the exact artifacts tested are identifiable later.

- [ ] **Step 6: Verify strict load against the real checkpoints**

```bash
conda activate hydra-mps
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
python -c "
from pathlib import Path
from tools.vitpose.external_ckpt.model import load_external_checkpoint
for name, expect in [
    ('ViTPose_base_ant9kp_256x256.pth', 9),
    ('ViTPose_base_fly29kp_ImgAug_256x256.pth', 29),
]:
    _, k = load_external_checkpoint(Path('/tmp/vitpose_external') / name)
    print(name, k, 'OK' if k == expect else 'MISMATCH')
"
```

Expected: `... 9 OK` and `... 29 OK`.

**If strict load raises a `RuntimeError` about missing/unexpected keys**, this is the plan's key risk materialising. Do not switch to `strict=False`. Capture the exact missing/unexpected key lists, stop, and report them — the key diff tells us precisely how their fork differs from our reimplementation, and that is the finding.

**If `torch.load` raises `_pickle.UnpicklingError` instead**, that is a different and expected snag: mmpose checkpoints carry a `meta` dict alongside `state_dict` that can hold non-tensor objects, which `weights_only=True` refuses. Do **not** drop to `weights_only=False` as a reflex — that unpickles arbitrary code from a downloaded file. Instead read the blocked global named in the error; if it is benign (typically `numpy.core.multiarray._reconstruct` or `numpy.dtype`), allow exactly it and nothing else, at the top of `model.py`:

```python
import numpy as np
import torch.serialization

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.dtype])
```

Re-run and confirm the load succeeds. If the blocked global is an mmcv/mmpose class rather than a numpy primitive, stop and report it rather than allowlisting it.

- [ ] **Step 7: Run the probe on both species**

```bash
conda activate hydra-mps
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
python -m tools.vitpose.external_ckpt.cli --species ant \
  --ckpt /tmp/vitpose_external/ViTPose_base_ant9kp_256x256.pth \
  --out /tmp/vitpose_probe
python -m tools.vitpose.external_ckpt.cli --species fly \
  --ckpt /tmp/vitpose_external/ViTPose_base_fly29kp_ImgAug_256x256.pth \
  --out /tmp/vitpose_probe
ls -la /tmp/vitpose_probe
```

Expected: `ant_axis.png`, `ant_rot.png`, `fly_axis.png`, `fly_rot.png` plus four `*_confidence.txt` files.

Open each PNG and judge:
1. Do keypoints land on the animal at all, or scatter over background?
2. Is `rot` visibly better than `axis`? That answers the orientation question.
3. Are the crops well framed — animal filling a reasonable share of the tile, nothing clipped? If the animal is tiny or cut off, re-run with `--scale 1.5` and `--scale 3.0` and compare. Crop framing is a guess in this plan and is the most likely cause of otherwise-inexplicable bad output.
4. In the confidence tables, which keypoints have low medians? For the fly, leg tips being weak while body keypoints are strong is a meaningful and expected pattern.

- [ ] **Step 8: Write the findings note and commit**

Create `tools/vitpose/external_ckpt/FINDINGS.md` recording: the two SHA-256 values, the exact commands run, whether strict load succeeded, the `--scale` finally used, the axis-vs-rot verdict, the per-keypoint confidence tables, and a one-paragraph conclusion on whether these checkpoints are worth promoting to first-class support (which would mean parameterising `IMAGE_SIZE_WH` in `src/`, deliberately out of scope here).

```bash
make format
make lint
git add tools/vitpose/external_ckpt tests/test_vitpose_external_ckpt.py
git commit -m "feat(tools): external ViTPose checkpoint probe CLI and findings"
```

---

## Out of Scope

- Any change under `src/`, including parameterising `IMAGE_SIZE_WH`/`HEATMAP_SIZE_WH`. If the probe says these checkpoints are good, that becomes a separate plan.
- Quantitative PCK/OKS scoring. There is no ground truth in their keypoint schema for our data, and the user chose a visual pass.
- Flip-test augmentation and DARK decode. Their config enables `flip_test=True`, which we skip; it changes results only marginally and adds a left/right-swap table this probe does not need.
- Overlay video export. Contact sheets only, per the agreed scope.
