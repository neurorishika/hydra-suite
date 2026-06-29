from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ComputeRuntime = Literal[
    "cpu", "mps", "cuda", "onnx_cpu", "onnx_cuda", "onnx_coreml", "tensorrt"
]

CUDA_RUNTIMES: frozenset[str] = frozenset({"cuda", "onnx_cuda", "tensorrt"})
CPU_RUNTIMES: frozenset[str] = frozenset({"cpu", "mps", "onnx_cpu", "onnx_coreml"})


class InferenceConfigError(ValueError):
    pass


@dataclass
class OBBDirectConfig:
    model_path: str
    compute_runtime: ComputeRuntime = "cpu"
    confidence_floor: float = 1e-3
    confidence_threshold: float = 0.25


@dataclass
class OBBSequentialConfig:
    detect_model_path: str
    obb_model_path: str
    detect_compute_runtime: ComputeRuntime = "cpu"
    obb_compute_runtime: ComputeRuntime = "cpu"
    detect_confidence_threshold: float = 1e-3
    obb_confidence_threshold: float = 1e-3
    detect_image_size: int = 0
    crop_pad_ratio: float = 0.15
    min_crop_size_px: float = 64.0
    enforce_square_crop: bool = True
    stage2_image_size: int = 160
    stage2_batch_size: int | None = None


@dataclass
class OBBConfig:
    mode: Literal["direct", "sequential"] = "direct"
    direct: OBBDirectConfig | None = None
    sequential: OBBSequentialConfig | None = None
    target_classes: list[int] = field(default_factory=list)
    max_detections: int = 20
    # Cap on RAW detections per frame, applied at OBB extraction (sorted by
    # confidence descending, top-k) BEFORE size/aspect/IoU filtering. Mirrors
    # legacy ``_obb_geometry._raw_detection_cap`` (= 2 * MAX_TARGETS). 0 disables.
    raw_detection_cap: int = 0
    min_object_size: float = 0.0
    max_object_size: float = float("inf")
    # Aspect-ratio (major/minor) gate, applied during filtering. Mirrors legacy
    # ``_obb_geometry`` aspect filtering (ref_ar * min/max multiplier). Defaults
    # (0, inf) disable the gate.
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float = float("inf")
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.7  # legacy YOLO_IOU_THRESHOLD default


@dataclass
class HeadTailConfig:
    model_path: str
    compute_runtime: ComputeRuntime = "cpu"
    confidence_threshold: float = 0.5
    candidate_confidence_threshold: float | None = None
    batch_size: int = 64
    canonical_aspect_ratio: float = 2.0
    canonical_margin: float = 1.3


@dataclass
class CNNConfig:
    label: str
    model_path: str
    compute_runtime: ComputeRuntime = "cpu"
    confidence_threshold: float = 0.5
    batch_size: int = 64
    scoring_mode: Literal["atomic", "per_head_average"] = "atomic"
    match_bonus: float = 0.1
    mismatch_penalty: float = 0.3
    calibration_temperature: float = 1.0


@dataclass
class PoseYOLOConfig:
    model_path: str
    compute_runtime: ComputeRuntime = "cpu"
    confidence_threshold: float = 1e-4
    iou_threshold: float = 0.7
    max_detections_per_crop: int = 1
    batch_size: int = 64


@dataclass
class PoseSLEAPConfig:
    model_path: str
    compute_runtime: ComputeRuntime = "cpu"
    conda_env: str = "sleap"
    batch_size: int = 4
    max_instances: int = 1


@dataclass
class PoseConfig:
    backend: Literal["yolo", "sleap"] = "yolo"
    skeleton_file: str = ""
    yolo: PoseYOLOConfig | None = None
    sleap: PoseSLEAPConfig | None = None
    crop_padding: float = 0.1
    suppress_foreign_regions: bool = True
    background_color: tuple[int, int, int] = (0, 0, 0)
    anterior_keypoints: list[str] = field(default_factory=list)
    posterior_keypoints: list[str] = field(default_factory=list)
    ignore_keypoints: list[str] = field(default_factory=list)
    min_keypoint_confidence: float = 0.2
    min_valid_keypoints: int = 1
    overrides_headtail: bool = True


@dataclass
class AprilTagConfig:
    enabled: bool = False
    tag_family: str = "tag36h11"
    threads: int = 4
    max_hamming: int = 1
    decimate: float = 1.0
    blur: float = 0.8
    refine_edges: bool = True
    decode_sharpening: float = 0.25
    unsharp_kernel: tuple[int, int] = (5, 5)
    unsharp_sigma: float = 1.0
    unsharp_amount: float = 1.5
    contrast_factor: float = 1.5
    max_tag_id: int | None = None
    crop_padding: float = 0.1


@dataclass
class InferenceConfig:
    obb: OBBConfig
    headtail: HeadTailConfig | None = None
    cnn_phases: list[CNNConfig] = field(default_factory=list)
    pose: PoseConfig | None = None
    apriltag: AprilTagConfig = field(default_factory=AprilTagConfig)
    detection_batch_size: int = 1
    realtime: bool = False
    use_cache: bool = True
    cache_dir: str | None = None

    @staticmethod
    def from_json(path: str) -> "InferenceConfig":
        with open(path) as f:
            data = json.load(f)
        config = _dict_to_config(data)
        config._validate_runtime_consistency()
        return config

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(_config_to_dict(self), f, indent=2)

    def _collect_all_runtimes(self) -> set[str]:
        runtimes: set[str] = set()
        if self.obb.direct:
            runtimes.add(self.obb.direct.compute_runtime)
        if self.obb.sequential:
            runtimes.add(self.obb.sequential.detect_compute_runtime)
            runtimes.add(self.obb.sequential.obb_compute_runtime)
        if self.headtail:
            runtimes.add(self.headtail.compute_runtime)
        for phase in self.cnn_phases:
            runtimes.add(phase.compute_runtime)
        if self.pose:
            if self.pose.yolo:
                runtimes.add(self.pose.yolo.compute_runtime)
            if self.pose.sleap:
                runtimes.add(self.pose.sleap.compute_runtime)
        return runtimes

    def _validate_runtime_consistency(self) -> None:
        runtimes = self._collect_all_runtimes()
        uses_cuda = bool(runtimes & CUDA_RUNTIMES)
        uses_cpu = bool(runtimes & CPU_RUNTIMES)
        if uses_cuda and uses_cpu:
            raise InferenceConfigError(
                f"Cannot mix CUDA-group and CPU-group runtimes. "
                f"CUDA-group found: {runtimes & CUDA_RUNTIMES}, "
                f"CPU-group found: {runtimes & CPU_RUNTIMES}"
            )


# ── serialization helpers ─────────────────────────────────────────────────────


def _config_to_dict(config: InferenceConfig) -> dict[str, Any]:
    d = asdict(config)
    obb = d["obb"]
    if obb.get("max_object_size") == float("inf"):
        obb["max_object_size"] = None
    if obb.get("max_aspect_ratio") == float("inf"):
        obb["max_aspect_ratio"] = None
    return d


def _dict_to_config(d: dict[str, Any]) -> InferenceConfig:
    obb_d = d["obb"]
    if obb_d.get("max_object_size") is None:
        obb_d["max_object_size"] = float("inf")
    if obb_d.get("max_aspect_ratio") is None:
        obb_d["max_aspect_ratio"] = float("inf")

    direct = OBBDirectConfig(**obb_d["direct"]) if obb_d.get("direct") else None
    sequential = (
        OBBSequentialConfig(**obb_d["sequential"]) if obb_d.get("sequential") else None
    )
    obb = OBBConfig(
        mode=obb_d["mode"],
        direct=direct,
        sequential=sequential,
        target_classes=obb_d.get("target_classes", []),
        max_detections=obb_d.get("max_detections", 20),
        raw_detection_cap=obb_d.get("raw_detection_cap", 0),
        min_object_size=obb_d.get("min_object_size", 0.0),
        max_object_size=obb_d.get("max_object_size", float("inf")),
        min_aspect_ratio=obb_d.get("min_aspect_ratio", 0.0),
        max_aspect_ratio=obb_d.get("max_aspect_ratio", float("inf")),
        confidence_threshold=obb_d.get("confidence_threshold", 0.25),
        iou_threshold=obb_d.get("iou_threshold", 0.45),
    )

    ht_d = d.get("headtail")
    headtail = HeadTailConfig(**ht_d) if ht_d else None

    cnn_phases = [CNNConfig(**c) for c in d.get("cnn_phases", [])]

    pose_d = d.get("pose")
    pose = None
    if pose_d:
        yolo_d = pose_d.pop("yolo", None)
        sleap_d = pose_d.pop("sleap", None)
        bg = pose_d.get("background_color")
        if isinstance(bg, list):
            pose_d["background_color"] = tuple(bg)
        pose = PoseConfig(
            **pose_d,
            yolo=PoseYOLOConfig(**yolo_d) if yolo_d else None,
            sleap=PoseSLEAPConfig(**sleap_d) if sleap_d else None,
        )

    at_d = d.get("apriltag", {})
    if isinstance(at_d.get("unsharp_kernel"), list):
        at_d["unsharp_kernel"] = tuple(at_d["unsharp_kernel"])
    apriltag = AprilTagConfig(**at_d) if at_d else AprilTagConfig()

    return InferenceConfig(
        obb=obb,
        headtail=headtail,
        cnn_phases=cnn_phases,
        pose=pose,
        apriltag=apriltag,
        detection_batch_size=d.get("detection_batch_size", 1),
        realtime=d.get("realtime", False),
        use_cache=d.get("use_cache", True),
        cache_dir=d.get("cache_dir"),
    )
