"""Build a classifier's ONNX Runtime TensorRT session in a throwaway process.

Some TensorRT builder failures are not Python exceptions the caller can
recover from: a ViT-B/16 identity model on TRT 10 hit ``CUDA error 716
(misaligned address)`` inside the autotuner, which poisons the whole CUDA
context, so even the native-torch fallback in the same process then fails.
Building the engine here first isolates that risk: if this child dies, only
the child's CUDA context is lost. On success the engine is in ORT's on-disk
cache, so the parent's own session creation just loads it.

Run as ``python -m hydra_suite.core.individual.classification.trt_probe
<json>`` where the JSON is ``{"model": <onnx path>, "providers": [...]}``.
Exit status 0 means the session was built.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

_STDERR_TAIL_CHARS = 2000


@dataclass(frozen=True)
class TrtProbeResult:
    ok: bool
    returncode: int
    detail: str


def _jsonable_providers(providers: list) -> list:
    return [p if isinstance(p, str) else [p[0], dict(p[1])] for p in providers]


def probe_trt_session_in_child(
    model_path: str, providers: list, *, runner=subprocess.run
) -> TrtProbeResult:
    """Create the ORT session for *model_path* in a child; report success."""
    payload = json.dumps(
        {"model": str(model_path), "providers": _jsonable_providers(providers)}
    )
    proc = runner(
        [sys.executable, "-m", __name__, payload],
        capture_output=True,
        text=True,
    )
    detail = ((proc.stderr or "") + (proc.stdout or ""))[-_STDERR_TAIL_CHARS:]
    return TrtProbeResult(proc.returncode == 0, proc.returncode, detail.strip())


def main(argv: list[str]) -> int:
    args = json.loads(argv[0])
    providers = [
        p if isinstance(p, str) else (p[0], dict(p[1])) for p in args["providers"]
    ]

    from hydra_suite.runtime.onnx_providers import preload_ort_cuda_libraries

    preload_ort_cuda_libraries()
    import onnxruntime as ort

    ort.InferenceSession(str(args["model"]), providers=providers)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
