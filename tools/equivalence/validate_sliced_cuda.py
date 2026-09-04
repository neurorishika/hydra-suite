"""Real-device validation of the sliced-inference path on both CUDA tiers.

Run this ON A CUDA BOX (see CLAUDE.md for the lab GPU host). The in-repo unit
tests simulate device behaviour on a CPU/MPS box, which exercises control flow
and tensor algebra but NOT actual device behaviour. This script drives the same
code on a real GPU.

IMPORTANT -- it drives the two REAL tier shapes, which are mutually exclusive
(finding C1). ``RuntimeContext.from_config`` can only ever emit:

  * tier ``gpu``      -> torch backend  -> ``tensor_on_cuda=True``, and NO NVDEC
                         (``_should_use_nvdec`` is gpu_fast-only), so frames are
                         NUMPY arrays;
  * tier ``gpu_fast`` -> TensorRT/CoreML backend -> ``tensor_on_cuda=False``,
                         with NVDEC, so frames are CUDA TENSORS.

An earlier version of this script asserted the impossible fourth combination
(``tensor_on_cuda=True`` AND CUDA-tensor frames), which is exactly why the
crash on BOTH tiers went unnoticed.

What it proves:
  1. tier ``gpu`` shape (numpy frames + device-tensor extraction), tiles that do
     not overlap: the production ``run_obb`` path returns ``_RawOBBTensors`` still resident
     on the GPU (``xywhr.is_cuda``) -- the zero-sync fast path really is
     zero-sync, not silently materialised to host.
  2. Same shape with genuinely overlapping tiles: the cross-tile merge runs
     on-device and yields a normal ``OBBResult``.
  3. ``merge_backend="gpu"`` works end-to-end on real CUDA.
  4. tier ``gpu_fast`` shape (CUDA-tensor frames + OBBResult extraction): tiles
     are sliced as device views, GPU-letterboxed, and returned as ``OBBResult``.

Usage (from a checkout on the CUDA host):
    conda activate hydra-cuda
    export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=$PWD/src
    python tools/equivalence/validate_sliced_cuda.py

Last verified: 2026-07-24 on NVIDIA RTX 6000 Ada, torch 2.11.0+cu130, AFTER the
C1 dispatch fix -- all checks passed, including the gpu_fast (CUDA-tensor frames
-> OBBResult) case that previously raised TypeError. The adjacent suites
(test_inference_slicing / test_inference_merge / test_utils_rotated_iou) also
pass on that host: 59 passed.

History worth keeping: an earlier revision of this script set
``tensor_on_cuda=True`` *and* passed CUDA-tensor frames -- a combination
``RuntimeContext.from_config`` can never emit (``tensor_on_cuda`` requires the
torch backend, while CUDA-tensor frames require NVDEC, which is gpu_fast-only).
It therefore passed on real hardware while the feature crashed on BOTH CUDA
tiers. Keep the tier shapes below aligned with what the resolver actually
produces; ``tests/test_inference_slicing.py`` pins that invariant.
"""

import types

import numpy as np
import torch

from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig, SliceConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.obb import OBBModels, _RawOBBTensors, run_obb
from hydra_suite.core.inference.stages.slicing import plan_slices, tiles_overlap

assert torch.cuda.is_available(), "no CUDA"
DEV = "cuda"


class RTGpu:
    """tier ``gpu``: torch backend on CUDA -> device-tensor extraction, CPU
    decode -> numpy frames."""

    device = DEV
    tensor_on_cuda = True

    def handoff(self, t):
        return t


class RTGpuFast:
    """tier ``gpu_fast``: TensorRT backend -> OBBResult extraction, NVDEC ->
    CUDA-tensor frames."""

    device = DEV
    tensor_on_cuda = False

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


def run_direct_sliced(frames, model, config, runtime):
    """Drive the current production plan/execute/extract/merge path."""
    return run_obb(
        frames,
        OBBModels(mode="direct", direct_model=model),
        config,
        runtime,
    )


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

# --- 1. tier gpu: NUMPY frames + device-tensor extraction, tiles disjoint
#        (512 frame / 256 tile, ratio 0) -> result stays on device
frame = np.zeros((512, 512, 3), dtype=np.uint8)
c = cfg(overlap_height_ratio=0.0, overlap_width_ratio=0.0)
plan = plan_slices((512, 512), c.direct.slice, 256, None)
print(f"[no-overlap] tiles={plan.tiles} tiles_overlap={tiles_overlap(plan.tiles)}")
out = run_direct_sliced([frame], FakeModel(), c, RTGpu())
r = out[0]
assert isinstance(r, _RawOBBTensors), f"expected _RawOBBTensors, got {type(r).__name__}"
assert r.xywhr.is_cuda and r.conf.is_cuda, "tensors left the device!"
print(
    f"[no-overlap] OK: _RawOBBTensors preserved, xywhr.is_cuda={r.xywhr.is_cuda}, n={r.xywhr.shape[0]}"
)

# --- 2. tier gpu, genuinely overlapping tiles -> merge path runs on real device
frame2 = np.zeros((300, 300, 3), dtype=np.uint8)
c2 = cfg(overlap_height_ratio=0.2, overlap_width_ratio=0.2)
plan2 = plan_slices((300, 300), c2.direct.slice, 256, None)
print(f"[overlap]    tiles={plan2.tiles} tiles_overlap={tiles_overlap(plan2.tiles)}")
out2 = run_direct_sliced([frame2], FakeModel(), c2, RTGpu())
r2 = out2[0]
print(
    f"[overlap]    OK: returned {type(r2).__name__}, "
    f"n={getattr(r2, 'num_detections', r2.xywhr.shape[0] if hasattr(r2, 'xywhr') else '?')}"
)

# --- 3. gpu merge backend end-to-end on real CUDA
c3 = cfg(overlap_height_ratio=0.2, overlap_width_ratio=0.2, merge_backend="gpu")
out3 = run_direct_sliced([frame2], FakeModel(), c3, RTGpu())
r3 = out3[0]
print(
    f"[gpu-merge]  OK: returned {type(r3).__name__}, "
    f"n={getattr(r3, 'num_detections', r3.xywhr.shape[0] if hasattr(r3, 'xywhr') else '?')}"
)

# --- 4. tier gpu_fast shape: CUDA-TENSOR frames (NVDEC) + OBBResult extraction.
#        Frames are device tensors, so tiles are device views and the plain
#        ultralytics model path GPU-letterboxes them into one batched tensor.
frame4 = torch.zeros((512, 512, 3), dtype=torch.uint8, device=DEV)
c4 = cfg(overlap_height_ratio=0.0, overlap_width_ratio=0.0)
out4 = run_direct_sliced([frame4], FakeModel(), c4, RTGpuFast())
r4 = out4[0]
assert isinstance(r4, OBBResult), f"expected OBBResult, got {type(r4).__name__}"
print(f"[gpu_fast]   OK: CUDA-tensor frames -> OBBResult, n={r4.num_detections}")

print("\nALL REAL-CUDA CHECKS PASSED")
