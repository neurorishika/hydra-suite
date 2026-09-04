"""
Batch size optimizer for YOLO detection based on available device memory.
"""

import logging

from .batch_policy import clamp_realtime_frame_batch_size, is_realtime_workflow
from .gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE, torch

logger = logging.getLogger(__name__)


class BatchOptimizer:
    """Optimize batch size for YOLO inference based on device capabilities."""

    def __init__(self, advanced_config=None):
        """
        Initialize batch optimizer.

        Args:
            advanced_config: Dictionary with memory allocation settings
        """
        self.advanced_config = advanced_config or {}
        self.device_type = None
        self.device_name = None
        self.available_memory = None

    def detect_device(self: object) -> object:
        """
        Detect available compute device and its memory.

        Returns:
            tuple: (device_type, device_name, available_memory_mb)
        """
        if torch is None:
            self.device_type = "cpu"
            self.device_name = "CPU (PyTorch not available)"
            self.available_memory = 0
            logger.warning("PyTorch not available - batching disabled")
            return

        if TORCH_CUDA_AVAILABLE:
            self.device_type = "cuda"
            self.device_name = torch.cuda.get_device_name(0)
            # Get available memory (free memory, not total)
            free_memory, total_memory = torch.cuda.mem_get_info(0)
            self.available_memory = free_memory / (1024**2)  # Convert to MB
            logger.info(f"CUDA device detected: {self.device_name}")
            logger.info(
                f"Available VRAM: {self.available_memory:.0f} MB / {total_memory / (1024 ** 2):.0f} MB"
            )

        elif MPS_AVAILABLE:
            self.device_type = "mps"
            self.device_name = "Apple Silicon (MPS)"
            # MPS uses one unified host-memory pool. Retain the raw available
            # amount here; `_auto_batch_size` applies the configured safety
            # fraction exactly once.
            try:
                import psutil

                available_memory = psutil.virtual_memory().available / (1024**2)
                self.available_memory = available_memory
                logger.info(f"MPS device detected: {self.device_name}")
                logger.info(f"Available unified memory: {self.available_memory:.0f} MB")
            except ImportError:
                # Fallback if psutil not available
                self.available_memory = 2048  # Conservative 2GB default
                logger.warning(
                    "psutil not available, using conservative 2GB estimate for MPS"
                )

        else:
            self.device_type = "cpu"
            self.device_name = "CPU"
            self.available_memory = 0  # CPU doesn't benefit from batching
            logger.info("CPU device detected - batching disabled")

        return (self.device_type, self.device_name, self.available_memory)

    def estimate_batch_size(
        self: object,
        frame_width: object,
        frame_height: object,
        model_name: object = "yolo26s-obb.pt",
    ) -> object:
        """
        Estimate optimal batch size for YOLO inference.

        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            model_name: YOLO model name (for memory estimation)

        Returns:
            int: Recommended batch size (1 if batching not recommended)
        """
        # Auto-detect device if not already done
        if self.device_type is None:
            self.detect_device()

        if is_realtime_workflow(
            self.advanced_config.get("tracking_realtime_mode", False),
            self.advanced_config.get("tracking_workflow_mode", "non_realtime"),
        ):
            logger.info("Realtime workflow: frame-level detection batch size = 1")
            return 1

        # CPU never batches
        if self.device_type == "cpu":
            logger.info("CPU device: batch size = 1 (no batching)")
            return 1

        # Check if batching is enabled in config
        if not self.advanced_config.get("enable_yolo_batching", True):
            logger.info("YOLO batching disabled in config: batch size = 1")
            return 1

        # Check for manual override
        manual = self._manual_batch_size()
        if manual is not None:
            return clamp_realtime_frame_batch_size(
                manual,
                realtime_enabled=self.advanced_config.get(
                    "tracking_realtime_mode", False
                ),
                workflow_mode=self.advanced_config.get(
                    "tracking_workflow_mode", "non_realtime"
                ),
            )

        return clamp_realtime_frame_batch_size(
            self._auto_batch_size(frame_width, frame_height, model_name),
            realtime_enabled=self.advanced_config.get("tracking_realtime_mode", False),
            workflow_mode=self.advanced_config.get(
                "tracking_workflow_mode", "non_realtime"
            ),
        )

    def _get_max_allowed_batch(self) -> int:
        """Return max allowed batch size considering TensorRT config."""
        tensorrt_max = self.advanced_config.get("tensorrt_max_batch_size", 64)
        if self.advanced_config.get("enable_tensorrt", False):
            return min(64, tensorrt_max)
        return 64

    def _manual_batch_size(self) -> object:
        """Return manual batch size if configured, else None."""
        batch_mode = self.advanced_config.get("yolo_batch_size_mode", "auto")
        if batch_mode != "manual":
            return None
        manual_size = self.advanced_config.get("yolo_manual_batch_size", 16)
        max_allowed = self._get_max_allowed_batch()
        clamped_size = max(1, min(manual_size, max_allowed))
        logger.info(f"Manual batch size: {manual_size} (clamped to {clamped_size})")
        return clamped_size

    def _auto_batch_size(self, frame_width, frame_height, model_name) -> int:
        """Estimate batch size from available GPU memory."""
        frame_memory_mb = (frame_width * frame_height * 3 * 4) / (1024**2)

        # Model memory overhead (rough estimates based on model size)
        model_overhead = {
            "yolo26n": 20,
            "yolo26s": 50,
            "yolo26m": 120,
            "yolo26l": 200,
        }
        overhead_mb = 50
        for key, value in model_overhead.items():
            if key in model_name.lower():
                overhead_mb = value
                break

        if self.device_type == "cuda":
            memory_fraction = self.advanced_config.get("cuda_memory_fraction", 0.7)
        else:
            memory_fraction = self.advanced_config.get("mps_memory_fraction", 0.3)

        usable_memory = self.available_memory * memory_fraction
        memory_per_frame_total = frame_memory_mb * 2.5
        available_for_batch = usable_memory - overhead_mb

        if available_for_batch <= 0:
            logger.warning(
                f"Insufficient memory for batching: {usable_memory:.0f} MB available, {overhead_mb} MB model overhead"
            )
            return 1

        estimated_batch = int(available_for_batch / memory_per_frame_total)
        safe_batch = int(estimated_batch * 0.8)
        max_allowed = self._get_max_allowed_batch()
        batch_size = max(1, min(safe_batch, max_allowed))

        logger.info("Batch size estimation:")
        logger.info(f"  Device: {self.device_type.upper()} ({self.device_name})")
        logger.info(f"  Frame size: {frame_width}x{frame_height}")
        logger.info(f"  Available memory: {self.available_memory:.0f} MB")
        logger.info(f"  Memory fraction: {memory_fraction * 100:.0f}%")
        logger.info(f"  Usable memory: {usable_memory:.0f} MB")
        logger.info(f"  Model overhead: {overhead_mb} MB")
        logger.info(f"  Memory per frame: {memory_per_frame_total:.1f} MB")
        logger.info(f"  Estimated batch size: {estimated_batch}")
        logger.info(f"  Safe batch size: {batch_size}")

        return batch_size

    def get_device_info(self: object) -> object:
        """
        Get human-readable device information.

        Returns:
            dict: Device information for display
        """
        if self.device_type is None:
            self.detect_device()

        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "available_memory_mb": self.available_memory,
            "batching_supported": self.device_type in ["cuda", "mps"],
        }
