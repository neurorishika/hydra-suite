"""Compute-runtime helpers and fallback stubs for PoseKit inference settings."""

try:
    from hydra_suite.runtime.compute_runtime import (
        CANONICAL_RUNTIMES,
        allowed_runtimes_for_pipelines,
        derive_pose_runtime_settings,
        infer_compute_runtime_from_legacy,
        runtime_label,
    )
except Exception:
    CANONICAL_RUNTIMES = [
        "cpu",
        "mps",
        "cuda",
        "onnx_coreml",
        "onnx_cpu",
        "onnx_cuda",
        "tensorrt",
    ]

    def runtime_label(runtime: str) -> str:
        """Return a human-readable uppercase label for a canonical runtime identifier."""
        return str(runtime or "cpu").strip().upper()

    def allowed_runtimes_for_pipelines(_pipelines):
        """Return the set of runtimes supported by all given pipeline keys (fallback: cpu only)."""
        return ["cpu"]

    def infer_compute_runtime_from_legacy(
        yolo_device, enable_tensorrt, pose_runtime_flavor
    ):
        """Derive the canonical compute-runtime string from legacy per-field settings."""
        if enable_tensorrt:
            return "tensorrt"
        flavor = str(pose_runtime_flavor or "").lower()
        if flavor.startswith("onnx_cuda"):
            return "onnx_cuda"
        if flavor.startswith("onnx_mps") or flavor.startswith("onnx_coreml"):
            return "onnx_coreml"
        if flavor.startswith("onnx"):
            return "onnx_cpu"
        return "cpu"

    def derive_pose_runtime_settings(compute_runtime: str, backend_family: str):
        """Translate a canonical compute-runtime key into backend-specific runtime settings dict."""
        rt = str(compute_runtime or "cpu").strip().lower()
        if rt == "onnx_coreml":
            return {"pose_runtime_flavor": "onnx_mps", "pose_sleap_device": "mps"}
        if rt == "onnx_cpu":
            return {"pose_runtime_flavor": "onnx_cpu", "pose_sleap_device": "cpu"}
        if rt in {"onnx_cuda"}:
            return {"pose_runtime_flavor": rt, "pose_sleap_device": "cuda:0"}
        if rt == "tensorrt":
            return {
                "pose_runtime_flavor": "tensorrt_cuda",
                "pose_sleap_device": "cuda:0",
            }
        return {"pose_runtime_flavor": rt or "cpu", "pose_sleap_device": rt or "cpu"}
