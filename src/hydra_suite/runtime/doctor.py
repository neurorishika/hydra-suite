"""``hydra doctor``: verify an installation, headless, on every platform.

Run as ``hydra doctor`` or ``python -m hydra_suite.runtime.doctor``. Never
creates a QApplication, never downloads anything. Exit status is non-zero iff a
check FAILS; warnings (optional features, an undownloaded gated checkpoint)
do not fail the run but always carry the command that fixes them.

Checks that can crash the interpreter (shared-library clashes, OpenMP aborts,
bad C extensions) run in a subprocess so one crash cannot hide the others.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as importlib_metadata
import io
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

#: Every GUI entry module; importing one proves its import graph resolves.
KIT_MODULES = (
    "hydra_suite.launcher.app",
    "hydra_suite.trackerkit.app",
    "hydra_suite.posekit.gui.main",
    "hydra_suite.classkit.app",
    "hydra_suite.detectkit.app",
    "hydra_suite.filterkit.app",
    "hydra_suite.refinekit.app",
)

OPENCV_DISTS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)

REINSTALL = "python install.py --recreate"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    fix: str = ""


def detect_tier() -> str:
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return "mps"
    try:
        import torch

        if torch.version.cuda:
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _subprocess_python(
    code: str, timeout: float = 300.0
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _last_line(text: str) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return lines[-1] if lines else "<no output>"


def _captured(fn: Callable[[], int]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            status = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"{type(exc).__name__}: {exc}")
            status = 1
    return status, buf.getvalue().strip()


def _installed(dist: str) -> Optional[str]:
    try:
        return importlib_metadata.version(dist)
    except importlib_metadata.PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python() -> Check:
    v = sys.version_info
    if v < (3, 11):
        return Check(
            "python", FAIL, platform.python_version(), "HYDRA needs Python >= 3.11"
        )
    return Check("python", OK, f"{platform.python_version()} ({sys.executable})")


def check_kit_imports() -> Check:
    code = (
        "import importlib, sys\n"
        f"mods = {KIT_MODULES!r}\n"
        "bad = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as e:\n"
        "        bad.append(f'{m}: {type(e).__name__}: {e}')\n"
        "print('\\n'.join(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    r = _subprocess_python(code)
    if r.returncode == 0:
        return Check(
            "kit imports", OK, f"{len(KIT_MODULES)} GUI modules import headless"
        )
    detail = r.stdout.strip() or _last_line(r.stderr)
    return Check("kit imports", FAIL, detail, REINSTALL)


def check_opencv() -> Check:
    present = {d: _installed(d) for d in OPENCV_DISTS}
    present = {d: v for d, v in present.items() if v}
    conda_owned = sorted(
        p.name.rsplit("-", 2)[0]
        for p in (Path(sys.prefix) / "conda-meta").glob("*opencv*.json")
    )
    if len(present) > 1 or conda_owned:
        owners = [f"{d} {v}" for d, v in present.items()] + [
            f"conda {c}" for c in conda_owned
        ]
        return Check(
            "opencv",
            FAIL,
            "multiple packages own cv2/: " + ", ".join(owners),
            f"rebuild the env: {REINSTALL}",
        )
    if not present:
        return Check("opencv", FAIL, "no opencv distribution installed", REINSTALL)
    ((dist, ver),) = present.items()
    return Check("opencv", OK, f"{dist} {ver}")


def check_openmp() -> Check:
    if platform.system() != "Darwin":
        return Check("openmp", SKIP, "only macOS ships conflicting libomp copies")
    from hydra_suite.runtime.macos_libomp import IMPORT_ORDERS

    for code in IMPORT_ORDERS:
        r = _subprocess_python(code)
        if r.returncode != 0:
            return Check(
                "openmp",
                FAIL,
                f"`{code}` aborts: {_last_line(r.stderr)}",
                "python -m hydra_suite.runtime.macos_libomp",
            )
    return Check("openmp", OK, "one OpenMP runtime maps in every import order")


def check_torch(tier: str) -> List[Check]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return [Check("torch", FAIL, f"import failed: {exc}", REINSTALL)]
    build = torch.version.cuda or "cpu"
    base = f"torch {torch.__version__} (build: {build})"
    if tier == "mps":
        if torch.backends.mps.is_available():
            return [Check("torch", OK, base + ", MPS available")]
        return [
            Check(
                "torch",
                FAIL,
                base + ", MPS NOT available",
                "needs macOS 12.3+ on Apple Silicon",
            )
        ]
    if tier == "cuda":
        from hydra_suite.runtime import cuda_checks

        out: List[Check] = []
        for name, fn in cuda_checks.CHECKS:
            status, text = _captured(getattr(cuda_checks, fn))
            out.append(
                Check(
                    name,
                    OK if status == 0 else FAIL,
                    _last_line(text) if status == 0 else text,
                    (
                        ""
                        if status == 0
                        else "python install.py --recreate [--cuda 12|13]"
                    ),
                )
            )
            if status != 0:
                break  # the first failure explains the rest
        return out
    if torch.version.cuda and not torch.cuda.is_available():
        return [
            Check(
                "torch",
                WARN,
                base + " is a CUDA build on a CPU tier (larger than needed)",
                REINSTALL,
            )
        ]
    return [Check("torch", OK, base)]


def check_onnxruntime(tier: str) -> Check:
    if tier == "cuda":
        return Check("onnxruntime", SKIP, "covered by the CUDA checks")
    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
    except Exception as exc:  # noqa: BLE001
        return Check("onnxruntime", FAIL, f"import failed: {exc}", REINSTALL)
    if tier == "mps" and "CoreMLExecutionProvider" not in providers:
        return Check("onnxruntime", WARN, f"no CoreML provider: {providers}", REINSTALL)
    return Check("onnxruntime", OK, f"{ort.__version__}: {', '.join(providers)}")


def check_coreml(tier: str) -> Check:
    if tier != "mps":
        return Check("coremltools", SKIP, "MPS tier only")
    version = _installed("coremltools")
    if not version:
        return Check(
            "coremltools",
            FAIL,
            "not installed (gpu_fast export on Apple Silicon)",
            REINSTALL,
        )
    return Check("coremltools", OK, version)


def check_sam2() -> Check:
    r = _subprocess_python("import sam2.build_sam")
    if r.returncode != 0:
        return Check("sam2", FAIL, _last_line(r.stderr), REINSTALL)
    return Check(
        "sam2", OK, f"sam2 {_installed('sam2') or '?'} (weights download on first use)"
    )


def check_sam3_inference() -> List[Check]:
    from hydra_suite.core.inference.semantic import checkpoints

    deps = checkpoints.probe_dependencies()
    if not deps.usable:
        return [
            Check(
                "sam3 inference",
                FAIL,
                deps.reason,
                "python install.py --only clip  (or a full re-run)",
            )
        ]
    out = [Check("sam3 inference", OK, "ultralytics SAM3 + CLIP + ftfy importable")]
    ckpt = checkpoints.probe_checkpoint()
    if ckpt.usable:
        out.append(
            Check("sam3 checkpoint", OK, str(checkpoints.checkpoint_path("sam3")))
        )
    else:
        out.append(
            Check(
                "sam3 checkpoint",
                WARN,
                f"not downloaded (~{checkpoints.CHECKPOINT_SIZE_GB:.2f} GB, licence-gated)",
                "accept the licence at https://huggingface.co/facebook/sam3, then run "
                "`hf auth login` (or set HF_TOKEN); DetectKit downloads it on first use",
            )
        )
    return out


def check_hf_token() -> Check:
    """A stored-but-invalid HF token 401s even PUBLIC downloads (SAM2)."""
    try:
        from huggingface_hub import get_token, whoami
        from huggingface_hub.errors import HfHubHTTPError
    except Exception:  # noqa: BLE001
        return Check("hf token", SKIP, "huggingface_hub not importable")
    token = get_token()
    if not token:
        return Check(
            "hf token",
            SKIP,
            "not logged in (needed only for the gated SAM3 weights)",
            "hf auth login",
        )
    try:
        whoami(token=token)
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 401:
            return Check(
                "hf token",
                WARN,
                "the stored Hugging Face token is invalid or expired",
                "hf auth login  (or: hf auth logout)",
            )
        return Check("hf token", SKIP, f"could not verify: {exc}")
    except Exception as exc:  # noqa: BLE001 - offline etc.
        return Check("hf token", SKIP, f"could not verify (offline?): {exc}")
    return Check("hf token", OK, "valid")


def check_apriltag() -> Check:
    r = _subprocess_python("import apriltag")
    if r.returncode != 0:
        return Check(
            "apriltag",
            WARN,
            "not importable (AprilTag identity unavailable)",
            "python install.py --only apriltag",
        )
    return Check("apriltag", OK, "AprilTag fork importable")


def check_ffmpeg() -> Check:
    from hydra_suite.utils.ffmpeg import ffmpeg_exe

    exe = ffmpeg_exe()
    if not exe:
        return Check(
            "ffmpeg",
            WARN,
            "no ffmpeg binary (video crop export unavailable)",
            REINSTALL,
        )
    return Check("ffmpeg", OK, exe)


def check_sam3_training(required: bool) -> Check:
    try:
        from hydra_suite.training.sam3_lora.availability import (
            probe_sam3_training_availability,
        )

        avail = probe_sam3_training_availability()
    except Exception as exc:  # noqa: BLE001
        status = FAIL if required else SKIP
        return Check("sam3 training", status, f"probe failed: {exc}")
    if avail.usable:
        return Check("sam3 training", OK, "hydra-sam3 sidecar ready")
    status = FAIL if required else SKIP
    reason = (avail.reason.splitlines() or [""])[0]
    return Check(
        "sam3 training",
        status,
        f"optional sidecar: {reason}",
        "python install.py --with-sam3-train  (CUDA only)",
    )


def _conda_env_names() -> List[str]:
    from hydra_suite.utils.conda_utils import run_conda

    try:
        r = run_conda(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return [Path(p).name for p in json.loads(r.stdout).get("envs", [])]
    except Exception:  # noqa: BLE001 - no conda, or unparsable output
        return []


def check_sleap() -> Check:
    exists = "sleap" in _conda_env_names()
    if exists:
        return Check("sleap", OK, "sleap sidecar env present")
    return Check(
        "sleap",
        SKIP,
        "optional sidecar not installed",
        "python install.py --with-sleap",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_checks(
    tier: Optional[str] = None, require_sam3_train: bool = False, quick: bool = False
) -> List[Check]:
    tier = tier or detect_tier()
    checks: List[Check] = [check_python()]
    if not quick:
        checks.append(check_kit_imports())
    checks += [check_opencv(), check_openmp()]
    checks += check_torch(tier)
    checks += [check_onnxruntime(tier), check_coreml(tier), check_sam2()]
    checks += check_sam3_inference()
    checks.append(check_hf_token())
    checks += [
        check_apriltag(),
        check_ffmpeg(),
        check_sam3_training(require_sam3_train),
        check_sleap(),
    ]
    return checks


_BADGE = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[ -- ]"}


def render(checks: List[Check], tier: str) -> str:
    lines = [f"hydra doctor -- tier: {tier}, {platform.system()} {platform.machine()}"]
    for c in checks:
        lines.append(f"{_BADGE[c.status]} {c.name}: {c.detail}")
        if c.fix and c.status in (WARN, FAIL):
            lines.append(f"         fix: {c.fix}")
    fails = sum(c.status == FAIL for c in checks)
    warns = sum(c.status == WARN for c in checks)
    lines.append(
        f"\n{fails} failed, {warns} warning(s)."
        if fails or warns
        else "\nAll checks passed."
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="hydra doctor", description="Verify the HYDRA installation."
    )
    p.add_argument(
        "--tier", choices=["cpu", "mps", "cuda"], help="expected tier (default: detect)"
    )
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument(
        "--require-sam3-train",
        action="store_true",
        help="fail if the SAM3 training sidecar is unusable",
    )
    p.add_argument(
        "--quick", action="store_true", help="skip the (slow) GUI-module import check"
    )
    args = p.parse_args(argv)
    tier = args.tier or detect_tier()
    checks = run_checks(tier, args.require_sam3_train, args.quick)
    if args.json:
        print(
            json.dumps({"tier": tier, "checks": [asdict(c) for c in checks]}, indent=2)
        )
    else:
        print(render(checks, tier))
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
