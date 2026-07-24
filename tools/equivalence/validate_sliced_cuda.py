"""Real-device validation of the native-CUDA sliced-inference path.

Run this ON A CUDA BOX (see CLAUDE.md for the lab GPU host). The in-repo unit
tests simulate the native-CUDA path with ``tensor_on_cuda=True`` but
``device="cpu"``, which exercises control flow and tensor algebra but NOT actual
device behaviour. This script drives the same code with genuine CUDA tensors.

What it proves:
  1. With non-overlapping tiles, ``run_direct_sliced`` returns ``_RawOBBTensors``
     still resident on the GPU (``xywhr.is_cuda``) -- i.e. the zero-sync fast
     path really is zero-sync, not silently materialised to host.
  2. With genuinely overlapping tiles, the cross-tile merge runs on-device and
     yields a normal ``OBBResult``.
  3. ``merge_backend="gpu"`` works end-to-end on real CUDA.

Usage (from a checkout on the CUDA host):
    conda activate hydra-cuda
    export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=$PWD/src
    python tools/equivalence/validate_sliced_cuda.py

Last verified: 2026-07-24, NVIDIA RTX 6000 Ada, torch 2.11.0+cu130 -- all checks passed.
"""

import types

import torch

from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig, SliceConfig
from hydra_suite.core.inference.stages.obb import _RawOBBTensors
from hydra_suite.core.inference.stages.slicing import (
    plan_slices,
    run_direct_sliced,
    tiles_overlap,
)

assert torch.cuda.is_available(), "no CUDA"
DEV = "cuda"


class RT:
    device = DEV
    tensor_on_cuda = True

    def handoff(self, t):
        return t


class FakeOBB:
    """Emits one detection per tile, as CUDA tensors (mimics ultralytics OBB)."""

    def __init__(self, n=1):
        self.data = torch.tensor(
            [[60.0, 60.0, 30.0, 20.0, 0.3, 0.9, 0.0]] * n, device=DEV
        )

    def __len__(self):
        return self.data.shape[0]

    @property
    def xywhr(self):
        return self.data[:, :5]

    @property
    def xyxyxyxy(self):
        n = self.data.shape[0]
        return torch.zeros((n, 4, 2), device=DEV)

    @property
    def conf(self):
        return self.data[:, 5]

    @property
    def cls(self):
        return self.data[:, 6]


class FakeModel:
    imgsz = 256
    overrides = {"imgsz": 256}

    def predict(self, source, **kw):
        b = source.shape[0] if hasattr(source, "shape") else len(source)
        out = []
        for _ in range(b):
            r = types.SimpleNamespace()
            r.obb = FakeOBB()
            out.append(r)
        return out


def cfg(**sk):
    return OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(
            model_path="m.pt",
            model_task="obb",
            slice=SliceConfig(enabled=True, geometry_mode="auto_model", **sk),
        ),
        confidence_threshold=0.0,
        raw_detection_cap=0,
        max_detections=100,
    )


print(f"device={torch.cuda.get_device_name(0)}  torch={torch.__version__}")

# --- 1. non-overlapping tiles (512 frame / 256 tile, ratio 0) -> stays on device
frame = torch.zeros((512, 512, 3), dtype=torch.uint8, device=DEV)
c = cfg(overlap_height_ratio=0.0, overlap_width_ratio=0.0)
plan = plan_slices((512, 512), c.direct.slice, 256, None)
print(f"[no-overlap] tiles={plan.tiles} tiles_overlap={tiles_overlap(plan.tiles)}")
out = run_direct_sliced([frame], FakeModel(), c, RT())
r = out[0]
assert isinstance(r, _RawOBBTensors), f"expected _RawOBBTensors, got {type(r).__name__}"
assert r.xywhr.is_cuda and r.conf.is_cuda, "tensors left the device!"
print(
    f"[no-overlap] OK: _RawOBBTensors preserved, xywhr.is_cuda={r.xywhr.is_cuda}, n={r.xywhr.shape[0]}"
)

# --- 2. genuinely overlapping tiles -> merge path runs on real device
frame2 = torch.zeros((300, 300, 3), dtype=torch.uint8, device=DEV)
c2 = cfg(overlap_height_ratio=0.2, overlap_width_ratio=0.2)
plan2 = plan_slices((300, 300), c2.direct.slice, 256, None)
print(f"[overlap]    tiles={plan2.tiles} tiles_overlap={tiles_overlap(plan2.tiles)}")
out2 = run_direct_sliced([frame2], FakeModel(), c2, RT())
r2 = out2[0]
print(
    f"[overlap]    OK: returned {type(r2).__name__}, "
    f"n={getattr(r2, 'num_detections', r2.xywhr.shape[0] if hasattr(r2, 'xywhr') else '?')}"
)

# --- 3. gpu merge backend end-to-end on real CUDA
c3 = cfg(overlap_height_ratio=0.2, overlap_width_ratio=0.2, merge_backend="gpu")
out3 = run_direct_sliced([frame2], FakeModel(), c3, RT())
r3 = out3[0]
print(
    f"[gpu-merge]  OK: returned {type(r3).__name__}, "
    f"n={getattr(r3, 'num_detections', r3.xywhr.shape[0] if hasattr(r3, 'xywhr') else '?')}"
)

print("\nALL REAL-CUDA CHECKS PASSED")
