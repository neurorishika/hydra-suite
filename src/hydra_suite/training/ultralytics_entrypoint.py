"""Ultralytics CLI entrypoint with a safe MPS target-assignment fallback."""

from __future__ import annotations

import os
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


def _uuid_tokens(mask: str) -> list[str]:
    tokens = [token.strip() for token in mask.split(",") if token.strip()]
    if not tokens or not all(token.upper().startswith("GPU-") for token in tokens):
        return []
    return [token[4:].lower() for token in tokens]


def latch_cuda_visible_devices(torch_module: Any = None) -> bool:
    """Make the supervisor's GPU pin survive Ultralytics' device selection.

    Ultralytics 8.4.45 ``select_device`` overwrites ``CUDA_VISIBLE_DEVICES``
    with the requested ordinal; if CUDA is not yet initialised, ``device=0``
    then trains on PHYSICAL GPU 0 (reproduced on a 10-GPU box). Until now the
    pin survived only because importing ``hydra_suite.utils.gpu_utils`` calls
    ``torch.cuda.is_available()``, which happens to initialise the driver
    first. CUDA reads the mask once, at first initialisation; initialising
    here, while the pin is still in place, makes every later rewrite inert on
    purpose. When the mask names UUIDs, each visible device is checked against
    it so a regression fails loudly instead of training on someone else's GPU.
    """

    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None or not mask.strip():
        return False
    if torch_module is None:
        import torch as torch_module
    expected = _uuid_tokens(mask)
    if not torch_module.cuda.is_available():
        if expected:
            # Ultralytics would rewrite the mask to an ordinal and reach a GPU
            # the supervisor never admitted.
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES={mask} pins a GPU but torch sees none"
            )
        return False
    torch_module.cuda.init()
    if not expected:
        return True
    count = torch_module.cuda.device_count()
    observed = [
        str(torch_module.cuda.get_device_properties(index).uuid).lower()
        for index in range(count)
    ]
    matched = len(observed) == len(expected) and all(
        uuid.startswith(token) or token.startswith(uuid)
        for uuid, token in zip(observed, expected)
    )
    if not matched:
        raise RuntimeError(
            "CUDA device pin was not honoured: CUDA_VISIBLE_DEVICES="
            f"{mask} but torch sees {observed or 'no devices'}"
        )
    return True


def main() -> None:
    """Install the compatibility shim before dispatching the Ultralytics CLI."""
    latch_cuda_visible_devices()

    from ultralytics.cfg import entrypoint
    from ultralytics.utils import LOGGER
    from ultralytics.utils.tal import TaskAlignedAssigner

    from hydra_suite.training.ultralytics_scale_balance import (
        install_sahi_scale_balance_and_grouped_sampling,
    )

    if install_mps_task_aligned_assigner_fallback(TaskAlignedAssigner):
        LOGGER.info(
            "Hydra compatibility: MPS TaskAlignedAssigner will execute on CPU "
            "to avoid a PyTorch MPS indexing fault."
        )
    install_sahi_scale_balance_and_grouped_sampling()
    entrypoint()


if __name__ == "__main__":
    main()
