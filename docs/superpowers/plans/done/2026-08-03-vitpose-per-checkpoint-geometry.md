# ViTPose Per-Checkpoint Geometry — Implementation Plan (Slice 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ViTPose input resolution a per-checkpoint property instead of a process-wide constant, so models can be trained and run at a geometry that fits our animals rather than the COCO human portrait aspect — and so externally trained checkpoints load first-class.

**Architecture:** A frozen `PoseGeometry` value object is threaded as an explicit parameter (defaulting to today's `(192, 256)`) through transforms, decode, heads, dataset, export, and the backend. Heatmap size stops being an independent constant and derives as `image / 4`. A new bicubic `pos_embed` resizer removes the hard blocker that currently makes fine-tuning at any other size impossible. Checkpoints record `input_size: [H, W]`; the adapter infers it when absent.

**Tech Stack:** Python 3.13, PyTorch 2.11, OpenCV, NumPy, pytest. Conda env `hydra-mps`.

Spec: `docs/superpowers/specs/2026-08-03-vitpose-per-checkpoint-geometry-design.md`

## Global Constraints

- **The default must not change.** `DEFAULT_GEOMETRY = PoseGeometry((192, 256))`. Every existing behaviour, and the ~25 test files that hardcode 256x192 / 64x48, must keep passing **unmodified**. If you find yourself editing an existing test's expected numbers, stop — you have changed the default by accident.
- **Every new `geom` parameter is keyword-with-default, added at the END of the signature.** This preserves all existing positional call sites. Never insert a parameter in the middle.
- Geometry dimensions must be positive multiples of **32** (patch-16 needs 16; 32 matches ClassKit's existing snapping convention and keeps the heatmap divisible by 8).
- `heatmap_size_wh` is **always** `(W // 4, H // 4)` — derived, never stored, never passed separately.
- Serialized key is `input_size` with value `[H, W]` (height first), matching `core/individual/classification/backend.py:82-97`. Internally geometry is `(W, H)`. Convert only in `to_hw`/`from_hw`.
- Ambiguity is an error, never a guess. A wrong patch grid produces a plausible model that is silently wrong everywhere.
- Do not touch `types.py` or the yolo/sleap backends.
- Environment for every command:
  ```
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps
  export KMP_DUPLICATE_LIB_OK=TRUE
  export PYTHONPATH=<worktree>/src
  ```
  Run pytest from the worktree root. `make format` before each commit. Repo-wide `make format-check` fails on a pre-existing `src/hydra_suite/refinekit/gui/dialogs/merge_wizard.py` — do not touch it.

## Deviation from the spec, with rationale

Spec §5 says to put the geometry into `_vitpose_artifact_signature`. **Do not do this.** The signature is computed *before* the checkpoint is loaded, on every cache probe — so including geometry would force a 1 GB `torch.load` merely to decide whether a cached artifact is still valid.

It is also unnecessary. Geometry is now a deterministic function of the checkpoint file, and `path_fingerprint_token` (mtime + size, `core/individual/pose/artifacts.py:69-77`) already identifies that file. Fingerprint therefore implies geometry, and geometry adds no discriminating power.

The recipe-tag bump `vitpose-v1` -> `vitpose-v2` is still required and still in the plan, because the exporter's behaviour changes and every existing artifact must be rebuilt once. Task 5 records this reasoning in a code comment.

## File Structure

**Created:**
- `src/hydra_suite/core/individual/pose/vitpose/geometry.py` — the `PoseGeometry` value object and `DEFAULT_GEOMETRY`. Pure; no torch, no cv2.
- `src/hydra_suite/core/individual/pose/vitpose/pos_embed.py` — patch-grid resolution and bicubic `pos_embed` resizing. Torch only.
- `tests/test_vitpose_geometry.py`, `tests/test_vitpose_pos_embed.py`, `tests/test_vitpose_geometry_threading.py`, `tests/test_vitpose_external_geometry_e2e.py`

**Modified:** `vitpose/config.py`, `transforms.py`, `infer.py`, `heads.py`, `vitpose.py`, `adapter.py`, `export.py`, `backends/vitpose.py`, `training/{config,dataset,validate,train,model_setup}.py`

---

### Task 1: The `PoseGeometry` value object

**Files:**
- Create: `src/hydra_suite/core/individual/pose/vitpose/geometry.py`
- Modify: `src/hydra_suite/core/individual/pose/vitpose/config.py:11-12`
- Test: `tests/test_vitpose_geometry.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PoseGeometry` (frozen dataclass, field `image_size_wh: tuple[int, int]`; properties `heatmap_size_wh -> tuple[int,int]`, `patch_grid_hw -> tuple[int,int]`, `num_tokens -> int`, `aspect -> float`; methods `to_hw() -> list[int]`, classmethod `from_hw(hw) -> PoseGeometry`); `DEFAULT_GEOMETRY: PoseGeometry`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vitpose_geometry.py`:

```python
"""PoseGeometry: the per-checkpoint input/heatmap geometry value object."""

from __future__ import annotations

import pytest

from hydra_suite.core.individual.pose.vitpose.geometry import (
    DEFAULT_GEOMETRY,
    PoseGeometry,
)


def test_default_geometry_matches_the_historical_constants():
    # The whole plan rests on the default being unchanged.
    assert DEFAULT_GEOMETRY.image_size_wh == (192, 256)
    assert DEFAULT_GEOMETRY.heatmap_size_wh == (48, 64)


def test_heatmap_is_always_a_quarter_of_the_image():
    for wh in [(192, 256), (256, 256), (320, 320), (128, 192)]:
        g = PoseGeometry(wh)
        assert g.heatmap_size_wh == (wh[0] // 4, wh[1] // 4)


def test_patch_grid_is_image_over_sixteen_in_hw_order():
    assert PoseGeometry((192, 256)).patch_grid_hw == (16, 12)
    assert PoseGeometry((256, 256)).patch_grid_hw == (16, 16)


def test_num_tokens_includes_the_cls_slot():
    # Upstream ViTPose keeps the MAE cls slot, so pos_embed is patches + 1.
    assert PoseGeometry((192, 256)).num_tokens == 16 * 12 + 1 == 193
    assert PoseGeometry((256, 256)).num_tokens == 16 * 16 + 1 == 257


def test_aspect_is_width_over_height():
    assert PoseGeometry((192, 256)).aspect == pytest.approx(0.75)
    assert PoseGeometry((256, 256)).aspect == pytest.approx(1.0)


def test_serialization_round_trip_is_height_first():
    g = PoseGeometry((192, 256))
    assert g.to_hw() == [256, 192]
    assert PoseGeometry.from_hw([256, 192]) == g
    assert PoseGeometry.from_hw(g.to_hw()) == g


def test_from_hw_accepts_a_tuple_and_normalizes_to_tuple_field():
    g = PoseGeometry.from_hw((256, 256))
    assert isinstance(g.image_size_wh, tuple)
    assert g.image_size_wh == (256, 256)


def test_list_input_is_normalized_to_a_tuple_so_the_value_stays_hashable():
    g = PoseGeometry([256, 256])
    assert g.image_size_wh == (256, 256)
    assert hash(g) == hash(PoseGeometry((256, 256)))


@pytest.mark.parametrize("bad", [(192, 250), (200, 256), (0, 256), (192, 0), (-32, 256)])
def test_dimensions_must_be_positive_multiples_of_thirty_two(bad):
    with pytest.raises(ValueError, match="multiple of 32|positive"):
        PoseGeometry(bad)


def test_error_message_names_the_offending_dimension():
    with pytest.raises(ValueError, match="height"):
        PoseGeometry((192, 250))
    with pytest.raises(ValueError, match="width"):
        PoseGeometry((200, 256))


def test_from_hw_rejects_wrong_length():
    with pytest.raises(ValueError, match="two"):
        PoseGeometry.from_hw([256])


def test_geometry_is_frozen():
    g = PoseGeometry((192, 256))
    with pytest.raises(Exception):
        g.image_size_wh = (256, 256)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vitpose_geometry.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'hydra_suite.core.individual.pose.vitpose.geometry'`.

- [ ] **Step 3: Implement `geometry.py`**

```python
"""Per-checkpoint ViTPose input geometry.

The model input size used to be a process-wide constant pinned to COCO's human
portrait aspect (192x256, 0.75). Our animals arrive from OBB tracking roughly
square, so that aspect spent about a quarter of the pixel budget on padding.
This value object makes the geometry a property of the checkpoint instead.

The heatmap is DERIVED, never stored: ClassicHead is two stride-2 transposed
convolutions applied to the patch grid, so its output is always
image / 16 * 4 == image / 4. Keeping it as a second constant only created a
pair that had to agree by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

PATCH_SIZE = 16
HEATMAP_DIVISOR = 4
SIZE_MULTIPLE = 32


@dataclass(frozen=True)
class PoseGeometry:
    """Model input geometry as (width, height).

    Dimensions must be positive multiples of 32: patch-16 embedding needs 16,
    and 32 matches the snapping convention already used for classifier input
    sizes while keeping the heatmap dimension divisible by 8.
    """

    image_size_wh: tuple[int, int]

    def __post_init__(self) -> None:
        wh = tuple(int(v) for v in self.image_size_wh)
        if len(wh) != 2:
            raise ValueError(
                f"image_size_wh must have two entries (width, height); got {wh!r}"
            )
        for value, name in ((wh[0], "width"), (wh[1], "height")):
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}")
            if value % SIZE_MULTIPLE:
                raise ValueError(
                    f"{name} must be a multiple of {SIZE_MULTIPLE}; got {value}"
                )
        object.__setattr__(self, "image_size_wh", wh)

    @property
    def heatmap_size_wh(self) -> tuple[int, int]:
        w, h = self.image_size_wh
        return (w // HEATMAP_DIVISOR, h // HEATMAP_DIVISOR)

    @property
    def patch_grid_hw(self) -> tuple[int, int]:
        w, h = self.image_size_wh
        return (h // PATCH_SIZE, w // PATCH_SIZE)

    @property
    def num_tokens(self) -> int:
        """Patch count plus the MAE cls slot upstream keeps in pos_embed."""
        gh, gw = self.patch_grid_hw
        return gh * gw + 1

    @property
    def aspect(self) -> float:
        w, h = self.image_size_wh
        return w / h

    def to_hw(self) -> list[int]:
        """[H, W] -- height first, matching the classifier stack's `input_size`."""
        w, h = self.image_size_wh
        return [h, w]

    @classmethod
    def from_hw(cls, hw: Sequence[int]) -> "PoseGeometry":
        values = [int(v) for v in hw]
        if len(values) != 2:
            raise ValueError(f"input_size must have two entries [H, W]; got {hw!r}")
        h, w = values
        return cls((w, h))


DEFAULT_GEOMETRY = PoseGeometry((192, 256))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vitpose_geometry.py -v`
Expected: all pass.

- [ ] **Step 5: Derive the legacy constants from the default**

In `src/hydra_suite/core/individual/pose/vitpose/config.py`, replace lines 11-12:

```python
IMAGE_SIZE_WH: tuple[int, int] = (192, 256)
HEATMAP_SIZE_WH: tuple[int, int] = (48, 64)
```

with:

```python
from .geometry import DEFAULT_GEOMETRY  # noqa: E402  (placed with the other imports)

# Retained as the DEFAULT geometry so existing callers and tests keep working.
# Per-checkpoint geometry now flows through PoseGeometry; see geometry.py.
IMAGE_SIZE_WH: tuple[int, int] = DEFAULT_GEOMETRY.image_size_wh
HEATMAP_SIZE_WH: tuple[int, int] = DEFAULT_GEOMETRY.heatmap_size_wh
```

Put the `from .geometry import DEFAULT_GEOMETRY` with the module's other imports at the top, not inline.

- [ ] **Step 6: Confirm the whole existing ViTPose suite is untouched**

Run: `python -m pytest tests/ -k vitpose -q`
Expected: all pass, including `tests/test_vitpose_config.py::` asserting `IMAGE_SIZE_WH == (192, 256)` and `HEATMAP_SIZE_WH == (48, 64)`. If either fails you have changed the default — fix, do not edit the test.

- [ ] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/core/individual/pose/vitpose/geometry.py src/hydra_suite/core/individual/pose/vitpose/config.py tests/test_vitpose_geometry.py
git commit -m "feat(pose): PoseGeometry value object; derive legacy size constants"
```

---

### Task 2: Patch-grid resolution and `pos_embed` resizing

This removes the hard blocker: `load_finetune_init` currently cannot load a checkpoint whose `pos_embed` token count differs from the model's, so fine-tuning at a new size is impossible today.

**Files:**
- Create: `src/hydra_suite/core/individual/pose/vitpose/pos_embed.py`
- Test: `tests/test_vitpose_pos_embed.py`

**Interfaces:**
- Consumes: `PoseGeometry`, `DEFAULT_GEOMETRY` from Task 1.
- Produces: `resolve_patch_grid(num_patches: int, stored: PoseGeometry | None = None) -> tuple[int, int]` returning `(gh, gw)`; `resize_pos_embed(pos_embed: torch.Tensor, src_grid_hw: tuple[int,int], dst_grid_hw: tuple[int,int]) -> torch.Tensor`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vitpose_pos_embed.py`:

```python
"""Patch-grid recovery and pos_embed interpolation."""

from __future__ import annotations

import pytest
import torch

from hydra_suite.core.individual.pose.vitpose.geometry import PoseGeometry
from hydra_suite.core.individual.pose.vitpose.pos_embed import (
    resize_pos_embed,
    resolve_patch_grid,
)


def test_stored_geometry_wins_over_inference():
    # 256 patches could be 16x16 or 8x32; a stored geometry settles it.
    assert resolve_patch_grid(256, PoseGeometry((512, 128))) == (8, 32)


def test_stored_geometry_must_agree_with_the_token_count():
    with pytest.raises(ValueError, match="does not match"):
        resolve_patch_grid(192, PoseGeometry((256, 256)))


def test_perfect_square_resolves_to_a_square_grid():
    # The collaborator's checkpoints: 257 pos_embed tokens -> 256 patches.
    assert resolve_patch_grid(256) == (16, 16)


def test_default_aspect_resolves_the_upstream_vitpose_grid():
    # Every upstream ViTPose release: 193 tokens -> 192 patches -> 12x16.
    assert resolve_patch_grid(192) == (16, 12)


def test_unresolvable_count_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="cannot determine|input_size"):
        resolve_patch_grid(150)


def test_error_names_the_token_count_and_asks_for_input_size():
    with pytest.raises(ValueError) as exc:
        resolve_patch_grid(150)
    assert "150" in str(exc.value)
    assert "input_size" in str(exc.value)


def test_resize_is_identity_when_grids_match():
    pe = torch.randn(1, 193, 768)
    out = resize_pos_embed(pe, (16, 12), (16, 12))
    assert torch.equal(out, pe)


def test_resize_produces_the_target_token_count():
    pe = torch.randn(1, 193, 768)
    out = resize_pos_embed(pe, (16, 12), (16, 16))
    assert out.shape == (1, 257, 768)


def test_resize_preserves_the_cls_slot_exactly():
    pe = torch.randn(1, 193, 768)
    out = resize_pos_embed(pe, (16, 12), (16, 16))
    assert torch.equal(out[:, :1], pe[:, :1])


def test_resize_rejects_a_source_grid_that_contradicts_the_tensor():
    pe = torch.randn(1, 193, 768)
    with pytest.raises(ValueError, match="does not match"):
        resize_pos_embed(pe, (16, 16), (16, 12))


def test_resize_round_trip_approximately_recovers_a_smooth_field():
    # Bicubic up-then-down on a smooth field should be close to the original.
    gh, gw, dim = 16, 12, 8
    ramp = torch.linspace(0, 1, gh * gw).reshape(1, gh * gw, 1).repeat(1, 1, dim)
    pe = torch.cat([torch.zeros(1, 1, dim), ramp], dim=1)
    up = resize_pos_embed(pe, (gh, gw), (32, 24))
    back = resize_pos_embed(up, (32, 24), (gh, gw))
    assert torch.allclose(back, pe, atol=2e-2)


def test_resized_weights_load_into_a_model_at_the_target_geometry():
    from hydra_suite.core.individual.pose.vitpose.vitpose import build_vitpose

    src = build_vitpose("B", "classic", num_keypoints=9)  # default 192x256
    dst_geom = PoseGeometry((256, 256))
    dst = build_vitpose("B", "classic", num_keypoints=9, geom=dst_geom)

    state = dict(src.state_dict())
    state["backbone.pos_embed"] = resize_pos_embed(
        state["backbone.pos_embed"], (16, 12), dst_geom.patch_grid_hw
    )
    missing, unexpected = dst.load_state_dict(state, strict=False)
    assert not missing and not unexpected

    with torch.no_grad():
        out = dst(torch.zeros(1, 3, 256, 256))
    assert out.shape == (1, 9, 64, 64)
```

Note: the last test consumes `build_vitpose(..., geom=...)`, which Task 3 adds. Expect it to fail until Task 3 lands; that is intentional and is called out in Step 4.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vitpose_pos_embed.py -v`
Expected: collection error — no module named `pos_embed`.

- [ ] **Step 3: Implement `pos_embed.py`**

```python
"""Patch-grid recovery and bicubic pos_embed resizing.

A bare token count does not determine the patch grid -- 256 patches could be
16x16 or 8x32 -- so recovery follows a fixed order and RAISES rather than
guessing. A wrong grid does not fail loudly at load time; it produces a
plausible model that is subtly wrong everywhere.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .geometry import DEFAULT_GEOMETRY, PoseGeometry


def resolve_patch_grid(
    num_patches: int, stored: PoseGeometry | None = None
) -> tuple[int, int]:
    """Recover (grid_h, grid_w) from a patch count.

    Order: a stored geometry is authoritative; then a perfect square; then the
    default 0.75 aspect (which covers every upstream ViTPose release); then
    raise.
    """
    if num_patches <= 0:
        raise ValueError(f"num_patches must be positive; got {num_patches}")

    if stored is not None:
        gh, gw = stored.patch_grid_hw
        if gh * gw != num_patches:
            raise ValueError(
                f"stored geometry {stored.image_size_wh} implies a {gh}x{gw} patch "
                f"grid ({gh * gw} patches) but the checkpoint has {num_patches}; "
                "the recorded input_size does not match the weights"
            )
        return (gh, gw)

    root = math.isqrt(num_patches)
    if root * root == num_patches:
        return (root, root)

    # Default aspect: width/height == 0.75, so grid_h/grid_w == 4/3.
    gh_sq = num_patches * 4
    if gh_sq % 3 == 0:
        gh = math.isqrt(gh_sq // 3)
        if gh > 0 and gh % 4 == 0:
            gw = (gh * 3) // 4
            if gh * gw == num_patches:
                return (gh, gw)

    raise ValueError(
        f"cannot determine the patch grid for {num_patches} patches: it is "
        "neither square nor the default 0.75 aspect. Record an explicit "
        "input_size [H, W] in the checkpoint."
    )


def resize_pos_embed(
    pos_embed: torch.Tensor,
    src_grid_hw: tuple[int, int],
    dst_grid_hw: tuple[int, int],
) -> torch.Tensor:
    """Bicubically resample the patch grid of a (1, 1 + gh*gw, D) pos_embed.

    The leading cls slot is carried through untouched. Returns the input
    object unchanged when the grids already match, so the default path is
    bit-for-bit unaffected.
    """
    src_h, src_w = src_grid_hw
    expected = src_h * src_w + 1
    if pos_embed.shape[1] != expected:
        raise ValueError(
            f"pos_embed has {pos_embed.shape[1]} tokens which does not match the "
            f"declared {src_h}x{src_w} source grid ({expected} tokens with the "
            "cls slot)"
        )
    if src_grid_hw == dst_grid_hw:
        return pos_embed

    cls_token = pos_embed[:, :1]
    grid = pos_embed[:, 1:]
    dim = grid.shape[-1]
    grid = grid.reshape(1, src_h, src_w, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(
        grid.float(), size=dst_grid_hw, mode="bicubic", align_corners=False
    )
    grid = grid.permute(0, 2, 3, 1).reshape(1, dst_grid_hw[0] * dst_grid_hw[1], dim)
    return torch.cat([cls_token, grid.to(pos_embed.dtype)], dim=1)


def grid_for_state(
    pos_embed: torch.Tensor, stored: PoseGeometry | None = None
) -> tuple[int, int]:
    """Convenience: resolve the grid for a checkpoint's pos_embed tensor."""
    return resolve_patch_grid(int(pos_embed.shape[1]) - 1, stored)
```

`DEFAULT_GEOMETRY` is imported for documentation parity with the rest of the package; if flake8 reports it unused, remove the import rather than adding a noqa.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_vitpose_pos_embed.py -v`
Expected: all pass EXCEPT `test_resized_weights_load_into_a_model_at_the_target_geometry`, which fails with `TypeError: build_vitpose() got an unexpected keyword argument 'geom'`. That is expected — Task 3 adds it. Every other test must pass now.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/individual/pose/vitpose/pos_embed.py tests/test_vitpose_pos_embed.py
git commit -m "feat(pose): patch-grid recovery and bicubic pos_embed resizing"
```

---

### Task 3: Thread geometry through the inference core

**Files:**
- Modify: `vitpose/transforms.py:26-36` (`box2cs`), `:78-95` (`affine_matrix`), `:97-103` (`top_down_affine`)
- Modify: `vitpose/infer.py:17-28` (`preprocess_crop`), `:30-46` (`decode_and_project`)
- Modify: `vitpose/heads.py:36-62` (`SimpleHead`, `build_head`)
- Modify: `vitpose/vitpose.py:27-33` (`build_vitpose`)
- Test: `tests/test_vitpose_geometry_threading.py`

**Interfaces:**
- Consumes: `PoseGeometry`, `DEFAULT_GEOMETRY` (Task 1).
- Produces: `box2cs(box_xywh, geom=DEFAULT_GEOMETRY)`; `affine_matrix(center, scale, rot=0.0, geom=DEFAULT_GEOMETRY)`; `top_down_affine(img, center, scale, rot=0.0, geom=DEFAULT_GEOMETRY)`; `preprocess_crop(crop_bgr, geom=DEFAULT_GEOMETRY)`; `decode_and_project(heatmaps, centers, scales, geom=DEFAULT_GEOMETRY)`; `SimpleHead(embed_dim, num_keypoints, geom=DEFAULT_GEOMETRY)`; `build_head(kind, embed_dim, num_keypoints, geom=DEFAULT_GEOMETRY)`; `build_vitpose(variant, head, num_keypoints=17, geom=DEFAULT_GEOMETRY)`.

**Every new parameter goes LAST with a default.** No existing call site changes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vitpose_geometry_threading.py`:

```python
"""Geometry threading: non-default geometry must reach every stage."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hydra_suite.core.individual.pose.vitpose.geometry import (
    DEFAULT_GEOMETRY,
    PoseGeometry,
)
from hydra_suite.core.individual.pose.vitpose.heads import build_head
from hydra_suite.core.individual.pose.vitpose.infer import preprocess_crop
from hydra_suite.core.individual.pose.vitpose.transforms import box2cs, top_down_affine
from hydra_suite.core.individual.pose.vitpose.vitpose import build_vitpose

SQUARE = PoseGeometry((256, 256))


def test_box2cs_uses_the_geometry_aspect_not_the_default():
    box = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32)
    _, scale_default = box2cs(box)
    _, scale_square = box2cs(box, geom=SQUARE)
    # Default aspect 0.75 grows a square box's height; square aspect does not.
    assert scale_default[1] > scale_square[1]
    assert scale_square[0] == pytest.approx(scale_square[1])


def test_box2cs_default_argument_is_unchanged():
    box = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32)
    assert box2cs(box) == pytest.approx(box2cs(box, geom=DEFAULT_GEOMETRY))


def test_top_down_affine_warps_to_the_geometry_size():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    center, scale = box2cs(
        np.array([0.0, 0.0, 200.0, 200.0], dtype=np.float32), geom=SQUARE
    )
    out = top_down_affine(img, center, scale, geom=SQUARE)
    assert out.shape == (256, 256, 3)


def test_top_down_affine_default_is_still_192x256():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    center, scale = box2cs(np.array([0.0, 0.0, 200.0, 200.0], dtype=np.float32))
    assert top_down_affine(img, center, scale).shape == (256, 192, 3)


def test_preprocess_crop_emits_the_geometry_shaped_tensor():
    crop = np.zeros((120, 120, 3), dtype=np.uint8)
    chw, _, _ = preprocess_crop(crop, geom=SQUARE)
    assert chw.shape == (3, 256, 256)
    chw_default, _, _ = preprocess_crop(crop)
    assert chw_default.shape == (3, 256, 192)


def test_simple_head_follows_the_geometry():
    # SimpleHead used to hardcode (64, 48); it must now track the geometry.
    head = build_head("simple", 768, 9, geom=SQUARE)
    out = head(torch.zeros(1, 768, 16, 16))
    assert out.shape == (1, 9, 64, 64)


def test_simple_head_default_is_unchanged():
    head = build_head("simple", 768, 9)
    out = head(torch.zeros(1, 768, 16, 12))
    assert out.shape == (1, 9, 64, 48)


def test_classic_head_scales_naturally_with_the_token_grid():
    head = build_head("classic", 768, 9, geom=SQUARE)
    out = head(torch.zeros(1, 768, 16, 16))
    assert out.shape == (1, 9, 64, 64)


def test_build_vitpose_constructs_the_backbone_at_the_geometry():
    model = build_vitpose("B", "classic", num_keypoints=9, geom=SQUARE)
    assert model.backbone.pos_embed.shape == (1, 257, 768)
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 256, 256))
    assert out.shape == (1, 9, 64, 64)


def test_build_vitpose_default_is_unchanged():
    model = build_vitpose("B", "classic", num_keypoints=9)
    assert model.backbone.pos_embed.shape == (1, 193, 768)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vitpose_geometry_threading.py -v`
Expected: failures — `box2cs() got an unexpected keyword argument 'geom'`, etc.

- [ ] **Step 3: Thread geometry through `transforms.py`**

Add `from .geometry import DEFAULT_GEOMETRY, PoseGeometry` to the imports. Then:

`box2cs` (line 26) — replace the body's aspect line:

```python
def box2cs(
    box_xywh: np.ndarray, geom: PoseGeometry = DEFAULT_GEOMETRY
) -> tuple[np.ndarray, np.ndarray]:
    x, y, w, h = box_xywh[:4]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    aspect = geom.aspect
    ...unchanged...
```

`affine_matrix` — add the parameter LAST and read the size from it:

```python
def affine_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float = 0.0,
    geom: PoseGeometry = DEFAULT_GEOMETRY,
) -> np.ndarray:
    w, h = geom.image_size_wh
    ...unchanged...
```

`top_down_affine` — same pattern, forwarding `geom`:

```python
def top_down_affine(
    img: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    rot: float = 0.0,
    geom: PoseGeometry = DEFAULT_GEOMETRY,
) -> np.ndarray:
    w, h = geom.image_size_wh
    trans = affine_matrix(center, scale, rot, geom)
    return cv2.warpAffine(img, trans, (w, h), flags=cv2.INTER_LINEAR)
```

Leave the `IMAGE_SIZE_WH` import in place only if something still uses it; otherwise remove it so flake8 stays clean.

- [ ] **Step 4: Thread geometry through `infer.py`**

Replace the `HEATMAP_SIZE_WH` import with `from .geometry import DEFAULT_GEOMETRY, PoseGeometry`, then:

```python
def preprocess_crop(
    crop_bgr: np.ndarray, geom: PoseGeometry = DEFAULT_GEOMETRY
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = crop_bgr.shape[:2]
    box_xywh = np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32)
    center, scale = box2cs(box_xywh, geom=geom)
    warped = top_down_affine(crop_bgr, center, scale, rot=0.0, geom=geom)
    chw = normalize(warped)
    return chw, center, scale


def decode_and_project(
    heatmaps: torch.Tensor,
    centers: np.ndarray,
    scales: np.ndarray,
    geom: PoseGeometry = DEFAULT_GEOMETRY,
) -> tuple[np.ndarray, np.ndarray]:
    coords_t, maxvals_t = decode_udp_torch(heatmaps)
    coords = coords_t.detach().cpu().numpy()
    maxvals = maxvals_t.detach().cpu().numpy()
    out = np.empty_like(coords)
    for i in range(coords.shape[0]):
        out[i] = transform_preds(
            coords[i], centers[i], scales[i], geom.heatmap_size_wh
        )
    return out, maxvals
```

Also update `decode_and_project`'s docstring: it says `heatmaps: (B, K, 64, 48)` — make it `(B, K, geom.heatmap_size_wh[1], geom.heatmap_size_wh[0])`.

- [ ] **Step 5: Thread geometry through `heads.py`**

Replace `from .config import HEATMAP_SIZE_WH` with `from .geometry import DEFAULT_GEOMETRY, PoseGeometry`, then:

```python
class SimpleHead(nn.Module):
    """num_deconv_layers=0, upsample=4, final_conv_kernel=3.

    Upstream applies ReLU inside _transform_inputs, i.e. BEFORE the upsample.
    """

    def __init__(
        self,
        embed_dim: int,
        num_keypoints: int,
        geom: PoseGeometry = DEFAULT_GEOMETRY,
    ) -> None:
        super().__init__()
        self.deconv_layers = nn.Identity()
        self.final_layer = nn.Conv2d(embed_dim, num_keypoints, 3, 1, 1)
        self._heatmap_size_wh = geom.heatmap_size_wh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        w, h = self._heatmap_size_wh
        # Explicit size (not scale_factor): scale_factor traces to a Resize with
        # computed sizes and is the classic ONNX shape-mismatch source. Same
        # result here, exportable later. align_corners=False is upstream's.
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return self.final_layer(self.deconv_layers(x))


def build_head(
    kind: str,
    embed_dim: int,
    num_keypoints: int,
    geom: PoseGeometry = DEFAULT_GEOMETRY,
) -> nn.Module:
    if kind == "classic":
        return ClassicHead(embed_dim, num_keypoints)
    if kind == "simple":
        return SimpleHead(embed_dim, num_keypoints, geom)
    raise ValueError(f"unknown head kind: {kind!r} (expected 'classic'|'simple')")
```

`ClassicHead` is unchanged — two stride-2 deconvs scale with the token grid on their own.

Also update the module docstring line `Input (B, D, 16, 12) -> output (B, K, 64, 48).` to say those are the default geometry's shapes.

- [ ] **Step 6: Thread geometry through `build_vitpose`**

In `vitpose.py`, add `from .geometry import DEFAULT_GEOMETRY, PoseGeometry` and:

```python
def build_vitpose(
    variant: str,
    head: str,
    num_keypoints: int = 17,
    geom: PoseGeometry = DEFAULT_GEOMETRY,
) -> ViTPose:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (expected one of SBLH)")
    v = VARIANTS[variant]
    backbone = ViT(
        embed_dim=v.embed_dim,
        depth=v.depth,
        num_heads=v.num_heads,
        img_size_hw=(geom.image_size_wh[1], geom.image_size_wh[0]),
    )
    return ViTPose(backbone, build_head(head, v.embed_dim, num_keypoints, geom))
```

Note `ViT.img_size_hw` is **(H, W)** while `PoseGeometry.image_size_wh` is **(W, H)** — the swap above is deliberate and is the single place it happens.

- [ ] **Step 7: Run the new and the existing tests**

Run: `python -m pytest tests/test_vitpose_geometry_threading.py tests/test_vitpose_pos_embed.py -v`
Expected: all pass, including `test_resized_weights_load_into_a_model_at_the_target_geometry` which was failing after Task 2.

Run: `python -m pytest tests/ -k vitpose -q`
Expected: all pass, unmodified.

- [ ] **Step 8: Commit**

```bash
make format
git add src/hydra_suite/core/individual/pose/vitpose/ tests/test_vitpose_geometry_threading.py
git commit -m "feat(pose): thread PoseGeometry through transforms, infer, heads, builder"
```

---

### Task 4: Adapter — record and recover geometry

**Files:**
- Modify: `vitpose/adapter.py` (`FinetuneMeta`, add `_infer_geometry`, `load_finetuned_checkpoint`)
- Test: `tests/test_vitpose_adapter_geometry.py`

**Interfaces:**
- Consumes: `PoseGeometry`/`DEFAULT_GEOMETRY` (Task 1), `resolve_patch_grid` (Task 2), `build_vitpose(..., geom=)` (Task 3).
- Produces: `FinetuneMeta` gains `geometry: PoseGeometry`; `load_finetuned_checkpoint(path) -> tuple[ViTPose, FinetuneMeta]` builds at the recovered geometry.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vitpose_adapter_geometry.py`:

```python
"""Geometry recovery when loading a fine-tuned or external checkpoint."""

from __future__ import annotations

import pytest
import torch

from hydra_suite.core.individual.pose.vitpose.adapter import load_finetuned_checkpoint
from hydra_suite.core.individual.pose.vitpose.geometry import (
    DEFAULT_GEOMETRY,
    PoseGeometry,
)
from hydra_suite.core.individual.pose.vitpose.vitpose import build_vitpose

SQUARE = PoseGeometry((256, 256))


def _save(tmp_path, payload, name="ckpt.pt"):
    path = tmp_path / name
    torch.save(payload, path)
    return path


def test_stored_input_size_is_authoritative(tmp_path):
    model = build_vitpose("B", "classic", num_keypoints=9, geom=SQUARE)
    path = _save(
        tmp_path,
        {
            "model_state": model.state_dict(),
            "variant": "B",
            "num_keypoints": 9,
            "input_size": SQUARE.to_hw(),
        },
    )
    loaded, meta = load_finetuned_checkpoint(path)
    assert meta.geometry == SQUARE
    assert loaded.backbone.pos_embed.shape == (1, 257, 768)


def test_square_geometry_is_inferred_when_not_stored(tmp_path):
    # This is the external-checkpoint case: no input_size key at all.
    model = build_vitpose("B", "classic", num_keypoints=9, geom=SQUARE)
    path = _save(tmp_path, {"state_dict": model.state_dict()})
    loaded, meta = load_finetuned_checkpoint(path)
    assert meta.geometry == SQUARE
    assert meta.num_keypoints == 9
    with torch.no_grad():
        assert loaded(torch.zeros(1, 3, 256, 256)).shape == (1, 9, 64, 64)


def test_default_geometry_is_inferred_for_an_upstream_shaped_checkpoint(tmp_path):
    model = build_vitpose("B", "classic", num_keypoints=17)
    path = _save(tmp_path, {"state_dict": model.state_dict()})
    _, meta = load_finetuned_checkpoint(path)
    assert meta.geometry == DEFAULT_GEOMETRY


def test_stored_geometry_contradicting_the_weights_raises(tmp_path):
    model = build_vitpose("B", "classic", num_keypoints=9)  # 193 tokens
    path = _save(
        tmp_path,
        {
            "model_state": model.state_dict(),
            "variant": "B",
            "num_keypoints": 9,
            "input_size": SQUARE.to_hw(),  # claims 257 tokens
        },
    )
    with pytest.raises(ValueError, match="does not match"):
        load_finetuned_checkpoint(path)


def test_meta_still_carries_variant_head_and_keypoints(tmp_path):
    model = build_vitpose("B", "classic", num_keypoints=9, geom=SQUARE)
    path = _save(tmp_path, {"state_dict": model.state_dict()})
    _, meta = load_finetuned_checkpoint(path)
    assert (meta.variant, meta.head, meta.num_keypoints) == ("B", "classic", 9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vitpose_adapter_geometry.py -v`
Expected: `AttributeError: 'FinetuneMeta' object has no attribute 'geometry'`, and the square cases fail to load.

- [ ] **Step 3: Implement in `adapter.py`**

Add imports:

```python
from .geometry import DEFAULT_GEOMETRY, PoseGeometry
from .pos_embed import resolve_patch_grid
```

Extend the dataclass — `geometry` goes LAST with a default so existing positional constructions keep working:

```python
@dataclass(frozen=True)
class FinetuneMeta:
    variant: str
    head: str
    num_keypoints: int
    geometry: PoseGeometry = DEFAULT_GEOMETRY
```

Add the resolver:

```python
def _infer_geometry(
    state: Dict[str, torch.Tensor], stored_hw: object = None
) -> PoseGeometry:
    """Recover the checkpoint's input geometry.

    A stored input_size is authoritative and is cross-checked against the
    weights. Otherwise the patch grid is recovered from the pos_embed token
    count, which raises rather than guessing when ambiguous.
    """
    pe = state.get("backbone.pos_embed")
    if pe is None:
        raise CheckpointKeyError(
            "checkpoint has no backbone.pos_embed; cannot infer geometry"
        )
    num_patches = int(pe.shape[1]) - 1
    stored = PoseGeometry.from_hw(stored_hw) if stored_hw is not None else None
    gh, gw = resolve_patch_grid(num_patches, stored)
    if stored is not None:
        return stored
    return PoseGeometry((gw * 16, gh * 16))
```

In `load_finetuned_checkpoint`, after `state = _unwrap_state(blob)`:

```python
    stored_hw = blob.get("input_size") if isinstance(blob, dict) else None
    geometry = _infer_geometry(state, stored_hw)
```

and change the model construction from
`model = build_vitpose(variant, head, num_keypoints=num_keypoints)` to

```python
    model = build_vitpose(variant, head, num_keypoints=num_keypoints, geom=geometry)
```

and the return to

```python
    return model, FinetuneMeta(
        variant=variant,
        head=head,
        num_keypoints=num_keypoints,
        geometry=geometry,
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_vitpose_adapter_geometry.py tests/test_vitpose_adapter.py -v`
Expected: all pass, including the pre-existing adapter tests unmodified.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/individual/pose/vitpose/adapter.py tests/test_vitpose_adapter_geometry.py
git commit -m "feat(pose): recover per-checkpoint geometry in the ViTPose adapter"
```

---

### Task 5: Backend and export

**Files:**
- Modify: `vitpose/export.py:77`, `:240` (dummy inputs)
- Modify: `backends/vitpose.py:81-83` (`preferred_input_size`), `:123-142` (`predict_batch`), `:149` (recipe tag)
- Test: `tests/test_vitpose_backend_geometry.py`

**Interfaces:**
- Consumes: `FinetuneMeta.geometry` (Task 4), `preprocess_crop`/`decode_and_project` with `geom` (Task 3).
- Produces: `export_onnx(..., geom=DEFAULT_GEOMETRY)`, `export_coreml(..., geom=DEFAULT_GEOMETRY)`; `ViTPoseBackend._geom`; recipe tag `vitpose-v2`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vitpose_backend_geometry.py`:

```python
"""Backend geometry threading and the artifact recipe tag."""

from __future__ import annotations

import torch

from hydra_suite.core.individual.pose.backends.vitpose import (
    _VITPOSE_RECIPE_TAG,
    _vitpose_artifact_signature,
    ViTPoseBackend,
)
from hydra_suite.core.individual.pose.vitpose.geometry import PoseGeometry
from hydra_suite.core.individual.pose.vitpose.vitpose import build_vitpose

SQUARE = PoseGeometry((256, 256))


def test_recipe_tag_is_bumped_so_old_artifacts_rebuild_once():
    # Geometry changes the exported graph; every v1 artifact must be invalidated.
    assert _VITPOSE_RECIPE_TAG == "vitpose-v2"


def test_signature_carries_the_recipe_tag(tmp_path):
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    sig = _vitpose_artifact_signature(str(ckpt), "coreml")
    assert sig.startswith("vitpose-v2|coreml|")


def _write_square_ckpt(tmp_path):
    model = build_vitpose("B", "classic", num_keypoints=9, geom=SQUARE)
    path = tmp_path / "square.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "variant": "B",
            "num_keypoints": 9,
            "input_size": SQUARE.to_hw(),
        },
        path,
    )
    return path


def test_backend_adopts_the_checkpoint_geometry(tmp_path):
    backend = ViTPoseBackend(str(_write_square_ckpt(tmp_path)), device="cpu")
    assert backend._geom == SQUARE
    assert backend.preferred_input_size == 256


def test_backend_predicts_end_to_end_at_a_square_geometry(tmp_path):
    import numpy as np

    backend = ViTPoseBackend(str(_write_square_ckpt(tmp_path)), device="cpu")
    results = backend.predict_batch([np.zeros((80, 80, 3), dtype=np.uint8)])
    assert len(results) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vitpose_backend_geometry.py -v`
Expected: `assert 'vitpose-v1' == 'vitpose-v2'` and `AttributeError: ... '_geom'`.

- [ ] **Step 3: Parameterize the export dummies**

In `export.py`, add `from .geometry import DEFAULT_GEOMETRY, PoseGeometry` to the imports. Add a keyword parameter `geom: PoseGeometry = DEFAULT_GEOMETRY` at the **end** of both `export_onnx`'s and `export_coreml`'s signatures, and replace both occurrences of

```python
    w, h = IMAGE_SIZE_WH
```

with

```python
    w, h = geom.image_size_wh
```

(one at line 77 in `export_onnx`, one at line 240 in `export_coreml`). Remove the now-unused `IMAGE_SIZE_WH` import if flake8 flags it.

In `export_onnx`'s docstring, the note that the input is fixed at 256x192 "because pos_embed has no interpolation path" is now wrong — `pos_embed.py` provides one, and the shape is fixed per exported artifact rather than globally. Update that sentence.

- [ ] **Step 4: Thread geometry through the backend**

In `backends/vitpose.py`:

Bump the tag and record why geometry is not in the signature:

```python
# v2: input geometry became per-checkpoint, which changes the exported graph,
# so every v1 artifact must be rebuilt once. The geometry itself is NOT in the
# signature: it is a deterministic function of the checkpoint file, and
# path_fingerprint_token already identifies that file. Adding it would force a
# full torch.load on every cache probe for no discriminating power.
_VITPOSE_RECIPE_TAG = "vitpose-v2"
```

In `__init__`, after `self._meta = meta`, add:

```python
        self._geom = meta.geometry
```

Replace `preferred_input_size` (line 81-83):

```python
    @property
    def preferred_input_size(self) -> int:
        return max(self._geom.image_size_wh)  # the long side
```

In `predict_batch`, forward the geometry at both call sites:

```python
                chw, c, s = preprocess_crop(np.asarray(crop), geom=self._geom)
```

```python
            coords, maxvals = decode_and_project(
                heatmaps, np.stack(centers), np.stack(scales), geom=self._geom
            )
```

In `auto_export_vitpose_model`, pass the loaded geometry to the exporters:

```python
    model, meta = load_finetuned_checkpoint(model_path)
    model.eval()
    if runtime_flavor == "coreml":
        export_coreml(model, artifact, geom=meta.geometry)
    elif runtime_flavor == "tensorrt":
        onnx_path = model_path.with_suffix(".onnx")
        export_onnx(model, onnx_path, geom=meta.geometry)
        build_tensorrt_engine(onnx_path, artifact, fp16=False)
```

(the local was named `_meta`; rename it to `meta` since it is now used).

Also update the `_forward_torch` comment `# (B, K, 64, 48) on device` — that shape is now the default geometry's, not universal.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_vitpose_backend_geometry.py tests/test_vitpose_backend_native.py tests/test_vitpose_export.py -v`
Expected: all pass. `tests/test_vitpose_backend_native.py:28` asserts `preferred_input_size == 256`, which still holds for a default-geometry checkpoint.

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/core/individual/pose/vitpose/export.py src/hydra_suite/core/individual/pose/backends/vitpose.py tests/test_vitpose_backend_geometry.py
git commit -m "feat(pose): per-checkpoint geometry in the ViTPose backend and exporters"
```

---

### Task 6: Training path

**Files:**
- Modify: `training/config.py:8-26` (`_FIELDS`), `:29-45` (`RunConfig`), `:54+` (`validate_run_config`)
- Modify: `training/model_setup.py:19-32` (`build_finetune_model`), `:34-61` (`load_finetune_init`)
- Modify: `training/dataset.py:11-17`, `:80-97` (`CocoKeypointsDataset`)
- Modify: `training/validate.py:23-55` (`run_validation`)
- Modify: `training/train.py` (resolve geometry once, thread it, stamp `input_size`)
- Test: `tests/test_vitpose_training_geometry.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: `RunConfig.input_size: list[int] | None = None`; `build_finetune_model(variant, num_keypoints, drop_path, geom=DEFAULT_GEOMETRY)`; `load_finetune_init(model, ckpt_path, geom=DEFAULT_GEOMETRY)`; `CocoKeypointsDataset(..., geom=DEFAULT_GEOMETRY)`; `run_validation(model, loader, device, thresholds=(0.05, 0.1), geom=DEFAULT_GEOMETRY)`; trainer payload gains `"input_size"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vitpose_training_geometry.py`:

```python
"""Training-side geometry: config validation, pos_embed resize, checkpoint stamp."""

from __future__ import annotations

import pytest
import torch

from hydra_suite.core.individual.pose.vitpose.geometry import (
    DEFAULT_GEOMETRY,
    PoseGeometry,
)
from hydra_suite.core.individual.pose.vitpose.training.config import validate_run_config
from hydra_suite.core.individual.pose.vitpose.training.model_setup import (
    build_finetune_model,
    load_finetune_init,
)
from hydra_suite.core.individual.pose.vitpose.vitpose import build_vitpose

SQUARE = PoseGeometry((256, 256))


def _base_cfg(**over):
    cfg = {
        "init_checkpoint": "x.pth",
        "variant": "B",
        "num_keypoints": 9,
        "dataset_dir": "d",
        "output_dir": "o",
    }
    cfg.update(over)
    return cfg


def test_input_size_defaults_to_none():
    assert validate_run_config(_base_cfg()).input_size is None


def test_input_size_is_accepted_as_height_width():
    assert validate_run_config(_base_cfg(input_size=[256, 256])).input_size == [256, 256]


@pytest.mark.parametrize("bad", [[256], [256, 250], [0, 256], "256x256"])
def test_malformed_input_size_is_rejected(bad):
    with pytest.raises(ValueError, match="input_size"):
        validate_run_config(_base_cfg(input_size=bad))


def test_build_finetune_model_honours_geometry():
    model = build_finetune_model("B", 9, 0.1, geom=SQUARE)
    assert model.backbone.pos_embed.shape == (1, 257, 768)


def test_build_finetune_model_default_is_unchanged():
    model = build_finetune_model("B", 9, 0.1)
    assert model.backbone.pos_embed.shape == (1, 193, 768)


def test_finetune_init_resizes_pos_embed_across_geometries(tmp_path):
    # THE point of this slice: initialise a 256x256 model from 192x256 weights.
    pretrained = build_vitpose("B", "classic", num_keypoints=17)
    ckpt = tmp_path / "pre.pth"
    torch.save({"state_dict": pretrained.state_dict()}, ckpt)

    model = build_finetune_model("B", 9, 0.1, geom=SQUARE)
    load_finetune_init(model, ckpt, geom=SQUARE)  # must not raise

    with torch.no_grad():
        out = model(torch.zeros(1, 3, 256, 256))
    assert out.shape == (1, 9, 64, 64)


def test_finetune_init_same_geometry_still_works(tmp_path):
    pretrained = build_vitpose("B", "classic", num_keypoints=17)
    ckpt = tmp_path / "pre.pth"
    torch.save({"state_dict": pretrained.state_dict()}, ckpt)
    model = build_finetune_model("B", 9, 0.1)
    load_finetune_init(model, ckpt)
    assert model.backbone.pos_embed.shape == (1, 193, 768)


def test_finetune_init_leaves_final_layer_fresh_across_geometries(tmp_path):
    pretrained = build_vitpose("B", "classic", num_keypoints=17)
    ckpt = tmp_path / "pre.pth"
    torch.save({"state_dict": pretrained.state_dict()}, ckpt)
    model = build_finetune_model("B", 9, 0.1, geom=SQUARE)
    load_finetune_init(model, ckpt, geom=SQUARE)
    # K differs, so the final layer must NOT have been loaded.
    assert model.keypoint_head.final_layer.weight.shape[0] == 9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vitpose_training_geometry.py -v`
Expected: `unknown run.json keys: ['input_size']`, and `build_finetune_model() got an unexpected keyword argument 'geom'`.

- [ ] **Step 3: Add `input_size` to `RunConfig`**

In `training/config.py`: add `"input_size"` to the `_FIELDS` set, add the field to the dataclass **last** (it has a default, so it must follow the other defaulted fields):

```python
    resume_from: str | None = None
    input_size: list[int] | None = None
```

Add validation inside `validate_run_config`, before `return RunConfig(**d)`:

```python
    size = d.get("input_size")
    if size is not None:
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("input_size must be a two-element list [H, W]")
        try:
            PoseGeometry.from_hw(size)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid input_size {size!r}: {exc}") from exc
        d = dict(d)
        d["input_size"] = [int(size[0]), int(size[1])]
```

with `from ..geometry import PoseGeometry` added to the imports. `PoseGeometry.from_hw` already enforces positive multiples of 32, so the rule lives in exactly one place.

- [ ] **Step 4: Thread geometry through `model_setup.py`**

Add imports:

```python
from ..geometry import DEFAULT_GEOMETRY, PoseGeometry
from ..pos_embed import grid_for_state, resize_pos_embed
```

`build_finetune_model` gains the parameter last and passes it on:

```python
def build_finetune_model(
    variant: str,
    num_keypoints: int,
    drop_path: float,
    geom: PoseGeometry = DEFAULT_GEOMETRY,
) -> ViTPose:
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r} (expected one of {sorted(VARIANTS)})"
        )
    v = VARIANTS[variant]
    backbone = ViT(
        embed_dim=v.embed_dim,
        depth=v.depth,
        num_heads=v.num_heads,
        img_size_hw=(geom.image_size_wh[1], geom.image_size_wh[0]),
        drop_path_rate=drop_path,
    )
    head = build_head("classic", v.embed_dim, num_keypoints, geom)
    return ViTPose(backbone, head)
```

`load_finetune_init` gains the parameter and resizes `pos_embed` when the grids differ. Insert this immediately after `cleaned` is built and before `model.load_state_dict`:

```python
def load_finetune_init(
    model: ViTPose, ckpt_path: Path, geom: PoseGeometry = DEFAULT_GEOMETRY
) -> None:
    ...
    cleaned = { ... unchanged ... }

    pe = cleaned.get("backbone.pos_embed")
    if pe is not None:
        src_grid = grid_for_state(pe)
        dst_grid = geom.patch_grid_hw
        if src_grid != dst_grid:
            print(
                f"resizing pos_embed {src_grid} -> {dst_grid} for fine-tune init",
                flush=True,
            )
            cleaned["backbone.pos_embed"] = resize_pos_embed(pe, src_grid, dst_grid)

    try:
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
    ... unchanged ...
```

- [ ] **Step 5: Thread geometry through dataset, validate, and train**

`training/dataset.py` — remove the module-level `FEAT_STRIDE` and the `HEATMAP_SIZE_WH, IMAGE_SIZE_WH` import; add `from ..geometry import DEFAULT_GEOMETRY, PoseGeometry`. Give `CocoKeypointsDataset.__init__` a `geom: PoseGeometry = DEFAULT_GEOMETRY` parameter (last), and store:

```python
        self.geom = geom
        self._feat_stride = (
            np.array(geom.image_size_wh, np.float32) - 1.0
        ) / (np.array(geom.heatmap_size_wh, np.float32) - 1.0)
```

Then in `__getitem__` replace `FEAT_STRIDE` with `self._feat_stride`, `w_hm, h_hm = HEATMAP_SIZE_WH` with `w_hm, h_hm = self.geom.heatmap_size_wh`, and the `generate_udp_gaussian(..., HEATMAP_SIZE_WH, self.sigma)` call with `self.geom.heatmap_size_wh`. Also forward `geom=self.geom` into the `box2cs`, `affine_matrix`, and `top_down_affine` calls in that method.

`training/validate.py` — replace `from ..config import HEATMAP_SIZE_WH` with `from ..geometry import DEFAULT_GEOMETRY, PoseGeometry`; add `geom: PoseGeometry = DEFAULT_GEOMETRY` last on `run_validation`; use `geom.heatmap_size_wh` in the `transform_preds` call.

`training/train.py` — resolve the geometry once near the top of `train(cfg)`:

```python
    geom = (
        PoseGeometry.from_hw(cfg.input_size)
        if cfg.input_size is not None
        else DEFAULT_GEOMETRY
    )
```

Thread it into `build_finetune_model(...)`, `load_finetune_init(...)`, both `CocoKeypointsDataset(...)` constructions, and `run_validation(model, val_loader, device, geom=geom)`. Add the stamp to the checkpoint payload:

```python
        ckpt = {
            "model_state": model.state_dict(),
            "optim_state": opt.state_dict(),
            "variant": cfg.variant,
            "num_keypoints": cfg.num_keypoints,
            "input_size": geom.to_hw(),
            "epoch": epoch,
            "pck": p05,
            "sched_state": sched.state_dict(),
        }
```

Give `_write_val_overlays` a `geom` parameter (last) and use `geom.heatmap_size_wh` in its `transform_preds` call; pass `geom` at the call site.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_vitpose_training_geometry.py -v`
Expected: all pass.

Run: `python -m pytest tests/ -k vitpose -q`
Expected: all pass, with no existing test edited.

- [ ] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/core/individual/pose/vitpose/training/ tests/test_vitpose_training_geometry.py
git commit -m "feat(pose): per-checkpoint geometry through the ViTPose training path"
```

---

### Task 7: End-to-end acceptance against the real external checkpoint

This is the slice's actual acceptance criterion: the collaborator's 256x256 checkpoint must load through the **production** path with no special flags and reproduce the coordinates the probe tool already validated.

**Files:**
- Create: `tests/test_vitpose_external_geometry_e2e.py`

**Interfaces:**
- Consumes: `load_finetuned_checkpoint` (Task 4), `ViTPoseBackend` (Task 5), and the probe package `tools.vitpose.external_ckpt.*`.

- [ ] **Step 1: Write the test**

Create `tests/test_vitpose_external_geometry_e2e.py`:

```python
"""The external 256x256 checkpoint must load through the production path.

Skipped unless the 1 GB checkpoint is present; it is not in the repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

CKPT = Path(
    "/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker"
    "/.worktrees/vitpose_external/ViTPose_base_ant9kp_256x256.pth"
)

pytestmark = pytest.mark.skipif(
    not CKPT.exists(), reason="external ViTPose checkpoint not present"
)


def test_external_checkpoint_loads_with_square_geometry():
    from hydra_suite.core.individual.pose.vitpose.adapter import (
        load_finetuned_checkpoint,
    )
    from hydra_suite.core.individual.pose.vitpose.geometry import PoseGeometry

    model, meta = load_finetuned_checkpoint(CKPT)
    assert meta.geometry == PoseGeometry((256, 256))
    assert meta.num_keypoints == 9
    assert meta.head == "classic"
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 256, 256))
    assert out.shape == (1, 9, 64, 64)


def test_production_loader_rebuilds_the_same_model_as_the_probe():
    """The probe's standalone loader is the validated reference. Given the same
    input tensor, the production-loaded model must produce the same heatmaps.

    Compare at the HEATMAP, not at final coordinates: the two paths preprocess
    differently on purpose (the probe warps the crop straight to 256x256, while
    production applies box2cs with PADDING_FACTOR=1.25) and decode differently
    on purpose (mmpose-'default' quarter-offset vs UDP). Feeding both the same
    tensor isolates what this slice actually changed -- checkpoint loading and
    model construction -- from those deliberate differences.
    """
    from hydra_suite.core.individual.pose.vitpose.adapter import (
        load_finetuned_checkpoint,
    )
    from tools.vitpose.external_ckpt.model import load_external_checkpoint, preprocess

    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    batch = torch.from_numpy(preprocess(crop)[None]).float()

    probe_model, _ = load_external_checkpoint(CKPT)
    prod_model, meta = load_finetuned_checkpoint(CKPT)

    with torch.no_grad():
        probe_out = probe_model.eval()(batch)
        prod_out = prod_model.eval()(batch)

    assert prod_out.shape == probe_out.shape == (1, 9, 64, 64)
    assert torch.allclose(prod_out, probe_out, atol=1e-5)


def test_production_preprocess_uses_the_checkpoint_geometry():
    from hydra_suite.core.individual.pose.vitpose.adapter import (
        load_finetuned_checkpoint,
    )
    from hydra_suite.core.individual.pose.vitpose.infer import preprocess_crop

    _, meta = load_finetuned_checkpoint(CKPT)
    chw, _, _ = preprocess_crop(np.zeros((120, 120, 3), dtype=np.uint8), geom=meta.geometry)
    assert chw.shape == (3, 256, 256)
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_vitpose_external_geometry_e2e.py -v`
Expected: 3 passed (the checkpoint is present in this worktree).

If `test_production_loader_rebuilds_the_same_model_as_the_probe` fails, do **not** widen `atol`. Both models hold the same weights and see the same tensor, so the heatmaps should agree to floating-point noise; a real difference means the production path built a different model — different geometry, a different head, or a mis-sized `pos_embed`. Report it rather than accommodating it.

- [ ] **Step 3: Run the runtime-parity gate**

Run:
```bash
python -m pytest tests/ -k vitpose -q
python tools/equivalence/verify_vitpose_runtimes.py --help
```
Expected: the suite passes; the verify script's help prints (a full run needs crops and an exported artifact, which is a merge-time gate, not a per-task one).

- [ ] **Step 4: Commit**

```bash
make format
git add tests/test_vitpose_external_geometry_e2e.py
git commit -m "test(pose): external 256x256 checkpoint loads through the production path"
```

---

## Out of Scope (Slice 2, same branch, before merge)

Per the spec's Slice 2 notes: `measure_pose_geometry` in `src/hydra_suite/training/`, `measured_input_size` in the dataset manifest, and the PoseKit training-dialog control with an "Auto from dataset" action and a Rescale x0.25-4.0 knob. None of that is in this plan.

Also out of scope here: removing `preferred_input_size` from the `types.py` Protocol and the yolo/sleap backends; model-registry unification.
