"""Behavior-preservation net for the Gen-2 runtime consolidation.

Every (tier, platform, stage, artifact_available) -> ResolvedBackend mapping
is frozen here. No task in the consolidation may change these values.
"""

import itertools

from hydra_suite.runtime.resolver import PlatformInfo, RuntimeResolver

TIERS = ("cpu", "gpu", "gpu_fast")
STAGES = ("obb", "head_tail", "cnn", "yolo_pose", "sleap_pose", "bgsub")
PLATFORMS = {
    "none": PlatformInfo(has_cuda=False, has_mps=False),
    "cuda": PlatformInfo(has_cuda=True, has_mps=False),
    "mps": PlatformInfo(has_cuda=False, has_mps=True),
    "both": PlatformInfo(has_cuda=True, has_mps=True),
}


def _key(tier, plat, stage, artifact):
    return f"{tier}|{plat}|{stage}|artifact={artifact}"


def _resolve(tier, plat_name, stage, artifact):
    r = RuntimeResolver(tier, PLATFORMS[plat_name]).resolve(
        stage, artifact_available=lambda: artifact
    )
    return (r.backend, r.device, r.used_fallback)


# Snapshot: values captured from current (pre-refactor) resolver.
# Regenerated ONCE, then treated as frozen for the rest of the plan.
EXPECTED_GOLDEN = {
    "cpu|none|obb|artifact=True": ("torch", "cpu", False),
    "cpu|none|obb|artifact=False": ("torch", "cpu", False),
    "cpu|none|head_tail|artifact=True": ("torch", "cpu", False),
    "cpu|none|head_tail|artifact=False": ("torch", "cpu", False),
    "cpu|none|cnn|artifact=True": ("torch", "cpu", False),
    "cpu|none|cnn|artifact=False": ("torch", "cpu", False),
    "cpu|none|yolo_pose|artifact=True": ("torch", "cpu", False),
    "cpu|none|yolo_pose|artifact=False": ("torch", "cpu", False),
    "cpu|none|sleap_pose|artifact=True": ("torch", "cpu", False),
    "cpu|none|sleap_pose|artifact=False": ("torch", "cpu", False),
    "cpu|none|bgsub|artifact=True": ("torch", "cpu", False),
    "cpu|none|bgsub|artifact=False": ("torch", "cpu", False),
    "cpu|cuda|obb|artifact=True": ("torch", "cpu", False),
    "cpu|cuda|obb|artifact=False": ("torch", "cpu", False),
    "cpu|cuda|head_tail|artifact=True": ("torch", "cpu", False),
    "cpu|cuda|head_tail|artifact=False": ("torch", "cpu", False),
    "cpu|cuda|cnn|artifact=True": ("torch", "cpu", False),
    "cpu|cuda|cnn|artifact=False": ("torch", "cpu", False),
    "cpu|cuda|yolo_pose|artifact=True": ("torch", "cpu", False),
    "cpu|cuda|yolo_pose|artifact=False": ("torch", "cpu", False),
    "cpu|cuda|sleap_pose|artifact=True": ("torch", "cpu", False),
    "cpu|cuda|sleap_pose|artifact=False": ("torch", "cpu", False),
    "cpu|cuda|bgsub|artifact=True": ("torch", "cpu", False),
    "cpu|cuda|bgsub|artifact=False": ("torch", "cpu", False),
    "cpu|mps|obb|artifact=True": ("torch", "cpu", False),
    "cpu|mps|obb|artifact=False": ("torch", "cpu", False),
    "cpu|mps|head_tail|artifact=True": ("torch", "cpu", False),
    "cpu|mps|head_tail|artifact=False": ("torch", "cpu", False),
    "cpu|mps|cnn|artifact=True": ("torch", "cpu", False),
    "cpu|mps|cnn|artifact=False": ("torch", "cpu", False),
    "cpu|mps|yolo_pose|artifact=True": ("torch", "cpu", False),
    "cpu|mps|yolo_pose|artifact=False": ("torch", "cpu", False),
    "cpu|mps|sleap_pose|artifact=True": ("torch", "cpu", False),
    "cpu|mps|sleap_pose|artifact=False": ("torch", "cpu", False),
    "cpu|mps|bgsub|artifact=True": ("torch", "cpu", False),
    "cpu|mps|bgsub|artifact=False": ("torch", "cpu", False),
    "cpu|both|obb|artifact=True": ("torch", "cpu", False),
    "cpu|both|obb|artifact=False": ("torch", "cpu", False),
    "cpu|both|head_tail|artifact=True": ("torch", "cpu", False),
    "cpu|both|head_tail|artifact=False": ("torch", "cpu", False),
    "cpu|both|cnn|artifact=True": ("torch", "cpu", False),
    "cpu|both|cnn|artifact=False": ("torch", "cpu", False),
    "cpu|both|yolo_pose|artifact=True": ("torch", "cpu", False),
    "cpu|both|yolo_pose|artifact=False": ("torch", "cpu", False),
    "cpu|both|sleap_pose|artifact=True": ("torch", "cpu", False),
    "cpu|both|sleap_pose|artifact=False": ("torch", "cpu", False),
    "cpu|both|bgsub|artifact=True": ("torch", "cpu", False),
    "cpu|both|bgsub|artifact=False": ("torch", "cpu", False),
    "gpu|none|obb|artifact=True": ("torch", "cpu", True),
    "gpu|none|obb|artifact=False": ("torch", "cpu", True),
    "gpu|none|head_tail|artifact=True": ("torch", "cpu", True),
    "gpu|none|head_tail|artifact=False": ("torch", "cpu", True),
    "gpu|none|cnn|artifact=True": ("torch", "cpu", True),
    "gpu|none|cnn|artifact=False": ("torch", "cpu", True),
    "gpu|none|yolo_pose|artifact=True": ("torch", "cpu", True),
    "gpu|none|yolo_pose|artifact=False": ("torch", "cpu", True),
    "gpu|none|sleap_pose|artifact=True": ("torch", "cpu", True),
    "gpu|none|sleap_pose|artifact=False": ("torch", "cpu", True),
    "gpu|none|bgsub|artifact=True": ("torch", "cpu", True),
    "gpu|none|bgsub|artifact=False": ("torch", "cpu", True),
    "gpu|cuda|obb|artifact=True": ("torch", "cuda", False),
    "gpu|cuda|obb|artifact=False": ("torch", "cuda", False),
    "gpu|cuda|head_tail|artifact=True": ("torch", "cuda", False),
    "gpu|cuda|head_tail|artifact=False": ("torch", "cuda", False),
    "gpu|cuda|cnn|artifact=True": ("torch", "cuda", False),
    "gpu|cuda|cnn|artifact=False": ("torch", "cuda", False),
    "gpu|cuda|yolo_pose|artifact=True": ("torch", "cuda", False),
    "gpu|cuda|yolo_pose|artifact=False": ("torch", "cuda", False),
    "gpu|cuda|sleap_pose|artifact=True": ("torch", "cuda", False),
    "gpu|cuda|sleap_pose|artifact=False": ("torch", "cuda", False),
    "gpu|cuda|bgsub|artifact=True": ("torch", "cuda", False),
    "gpu|cuda|bgsub|artifact=False": ("torch", "cuda", False),
    "gpu|mps|obb|artifact=True": ("torch", "mps", False),
    "gpu|mps|obb|artifact=False": ("torch", "mps", False),
    "gpu|mps|head_tail|artifact=True": ("torch", "mps", False),
    "gpu|mps|head_tail|artifact=False": ("torch", "mps", False),
    "gpu|mps|cnn|artifact=True": ("torch", "mps", False),
    "gpu|mps|cnn|artifact=False": ("torch", "mps", False),
    "gpu|mps|yolo_pose|artifact=True": ("torch", "mps", False),
    "gpu|mps|yolo_pose|artifact=False": ("torch", "mps", False),
    "gpu|mps|sleap_pose|artifact=True": ("torch", "mps", False),
    "gpu|mps|sleap_pose|artifact=False": ("torch", "mps", False),
    "gpu|mps|bgsub|artifact=True": ("torch", "mps", False),
    "gpu|mps|bgsub|artifact=False": ("torch", "mps", False),
    "gpu|both|obb|artifact=True": ("torch", "cuda", False),
    "gpu|both|obb|artifact=False": ("torch", "cuda", False),
    "gpu|both|head_tail|artifact=True": ("torch", "cuda", False),
    "gpu|both|head_tail|artifact=False": ("torch", "cuda", False),
    "gpu|both|cnn|artifact=True": ("torch", "cuda", False),
    "gpu|both|cnn|artifact=False": ("torch", "cuda", False),
    "gpu|both|yolo_pose|artifact=True": ("torch", "cuda", False),
    "gpu|both|yolo_pose|artifact=False": ("torch", "cuda", False),
    "gpu|both|sleap_pose|artifact=True": ("torch", "cuda", False),
    "gpu|both|sleap_pose|artifact=False": ("torch", "cuda", False),
    "gpu|both|bgsub|artifact=True": ("torch", "cuda", False),
    "gpu|both|bgsub|artifact=False": ("torch", "cuda", False),
    "gpu_fast|none|obb|artifact=True": ("torch", "cpu", True),
    "gpu_fast|none|obb|artifact=False": ("torch", "cpu", True),
    "gpu_fast|none|head_tail|artifact=True": ("torch", "cpu", True),
    "gpu_fast|none|head_tail|artifact=False": ("torch", "cpu", True),
    "gpu_fast|none|cnn|artifact=True": ("torch", "cpu", True),
    "gpu_fast|none|cnn|artifact=False": ("torch", "cpu", True),
    "gpu_fast|none|yolo_pose|artifact=True": ("torch", "cpu", True),
    "gpu_fast|none|yolo_pose|artifact=False": ("torch", "cpu", True),
    "gpu_fast|none|sleap_pose|artifact=True": ("torch", "cpu", True),
    "gpu_fast|none|sleap_pose|artifact=False": ("torch", "cpu", True),
    "gpu_fast|none|bgsub|artifact=True": ("torch", "cpu", True),
    "gpu_fast|none|bgsub|artifact=False": ("torch", "cpu", True),
    "gpu_fast|cuda|obb|artifact=True": ("tensorrt", "cuda", False),
    "gpu_fast|cuda|obb|artifact=False": ("torch", "cuda", True),
    "gpu_fast|cuda|head_tail|artifact=True": ("tensorrt", "cuda", False),
    "gpu_fast|cuda|head_tail|artifact=False": ("torch", "cuda", True),
    "gpu_fast|cuda|cnn|artifact=True": ("tensorrt", "cuda", False),
    "gpu_fast|cuda|cnn|artifact=False": ("torch", "cuda", True),
    "gpu_fast|cuda|yolo_pose|artifact=True": ("tensorrt", "cuda", False),
    "gpu_fast|cuda|yolo_pose|artifact=False": ("torch", "cuda", True),
    "gpu_fast|cuda|sleap_pose|artifact=True": ("tensorrt", "cuda", False),
    "gpu_fast|cuda|sleap_pose|artifact=False": ("torch", "cuda", True),
    "gpu_fast|cuda|bgsub|artifact=True": ("torch", "cuda", True),
    "gpu_fast|cuda|bgsub|artifact=False": ("torch", "cuda", True),
    "gpu_fast|mps|obb|artifact=True": ("coreml", "mps", False),
    "gpu_fast|mps|obb|artifact=False": ("torch", "mps", True),
    "gpu_fast|mps|head_tail|artifact=True": ("coreml", "mps", False),
    "gpu_fast|mps|head_tail|artifact=False": ("torch", "mps", True),
    "gpu_fast|mps|cnn|artifact=True": ("coreml", "mps", False),
    "gpu_fast|mps|cnn|artifact=False": ("torch", "mps", True),
    "gpu_fast|mps|yolo_pose|artifact=True": ("coreml", "mps", False),
    "gpu_fast|mps|yolo_pose|artifact=False": ("torch", "mps", True),
    "gpu_fast|mps|sleap_pose|artifact=True": ("coreml", "mps", False),
    "gpu_fast|mps|sleap_pose|artifact=False": ("torch", "mps", True),
    "gpu_fast|mps|bgsub|artifact=True": ("torch", "mps", True),
    "gpu_fast|mps|bgsub|artifact=False": ("torch", "mps", True),
    "gpu_fast|both|obb|artifact=True": ("tensorrt", "cuda", False),
    "gpu_fast|both|obb|artifact=False": ("torch", "cuda", True),
    "gpu_fast|both|head_tail|artifact=True": ("tensorrt", "cuda", False),
    "gpu_fast|both|head_tail|artifact=False": ("torch", "cuda", True),
    "gpu_fast|both|cnn|artifact=True": ("tensorrt", "cuda", False),
    "gpu_fast|both|cnn|artifact=False": ("torch", "cuda", True),
    "gpu_fast|both|yolo_pose|artifact=True": ("tensorrt", "cuda", False),
    "gpu_fast|both|yolo_pose|artifact=False": ("torch", "cuda", True),
    "gpu_fast|both|sleap_pose|artifact=True": ("tensorrt", "cuda", False),
    "gpu_fast|both|sleap_pose|artifact=False": ("torch", "cuda", True),
    "gpu_fast|both|bgsub|artifact=True": ("torch", "cuda", True),
    "gpu_fast|both|bgsub|artifact=False": ("torch", "cuda", True),
}


def test_golden_table_is_stable():
    table = {}
    for tier, plat, stage, artifact in itertools.product(
        TIERS, PLATFORMS, STAGES, (True, False)
    ):
        table[_key(tier, plat, stage, artifact)] = _resolve(tier, plat, stage, artifact)
    # Snapshot: values captured from current (pre-refactor) resolver.
    # Regenerate ONCE now, then treat as frozen for the rest of the plan.
    assert table == EXPECTED_GOLDEN
