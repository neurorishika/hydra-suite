"""Ultralytics CLI entrypoint with a safe MPS target-assignment fallback."""

from __future__ import annotations

from typing import Any


def install_mps_task_aligned_assigner_fallback(assigner_cls: type) -> bool:
    """Run TaskAlignedAssigner on CPU when its inputs are on MPS.

    PyTorch's MPS indexing kernels can nondeterministically corrupt target
    assignment on dense detection and segmentation batches.  The model forward
    and loss remain on MPS; only the small assignment computation crosses to
    CPU, then its result is returned to the original device.
    """
    original_forward = assigner_cls.forward
    if getattr(original_forward, "_hydra_mps_cpu_fallback", False):
        return False

    def forward_with_mps_fallback(self, *args: Any, **kwargs: Any):
        pd_scores = args[0] if args else kwargs.get("pd_scores")
        device = getattr(pd_scores, "device", None)
        if getattr(device, "type", None) != "mps":
            return original_forward(self, *args, **kwargs)

        cpu_args = tuple(value.cpu() for value in args)
        cpu_kwargs = {key: value.cpu() for key, value in kwargs.items()}
        results = original_forward(self, *cpu_args, **cpu_kwargs)
        return tuple(value.to(device) for value in results)

    forward_with_mps_fallback._hydra_mps_cpu_fallback = True
    assigner_cls.forward = forward_with_mps_fallback
    return True


def main() -> None:
    """Install the compatibility shim before dispatching the Ultralytics CLI."""
    from ultralytics.cfg import entrypoint
    from ultralytics.utils import LOGGER
    from ultralytics.utils.tal import TaskAlignedAssigner

    if install_mps_task_aligned_assigner_fallback(TaskAlignedAssigner):
        LOGGER.info(
            "Hydra compatibility: MPS TaskAlignedAssigner will execute on CPU "
            "to avoid a PyTorch MPS indexing fault."
        )
    entrypoint()


if __name__ == "__main__":
    main()
