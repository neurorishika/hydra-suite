#!/usr/bin/env python3
"""Cross-platform installer for HYDRA Suite (Linux, macOS, Windows).

Run it with ANY Python >= 3.9 -- base conda, a system ``python3``, or ``py -3``
on Windows. It never imports ``hydra_suite`` and uses only the standard
library, because it runs before (and outside) the environment it builds.

    python install.py                      # detect platform, build/refresh env, run doctor
    python install.py --dry-run            # print the plan, change nothing
    python install.py --cuda 12            # override driver-based CUDA detection
    python install.py --with-sam3-train    # + SAM3 LoRA training sidecar (CUDA only)
    python install.py --with-sleap         # + SLEAP pose sidecar env
    python install.py --target current     # install into the running interpreter

Python dependencies are declared ONLY in pyproject.toml. This script decides
*which* extras and *which* torch build a platform gets, and performs the steps
pip cannot express: the torch wheel index, the CLIP and AprilTag git installs,
platform library fixups, and the optional sidecar envs.

See docs/getting-started/installation.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Pins. Bump deliberately and re-verify on every tier.
# ---------------------------------------------------------------------------

#: torch/torchvision are a matched release pair; bump both together. CUDA
#: builds carry a ``+cuXXX`` local tag that exists only on the PyTorch index,
#: which is what stops PyPI's default (CUDA 13) Linux wheel from being chosen
#: on a CUDA 12 host -- the silent ``torch.cuda.is_available() == False`` trap.
TORCH_VERSION = "2.11.0"
TORCHVISION_VERSION = "0.26.0"
PYTORCH_INDEX = "https://download.pytorch.org/whl"

#: First driver major that supports CUDA 13 user-space (``nvidia-smi`` reports
#: "CUDA Version: 13.x" from this branch on).
CUDA13_MIN_DRIVER = 580

CLIP_URL = "git+https://github.com/ultralytics/CLIP.git"
CLIP_REF = "a13192f8cb767260d7dfd98c843b0716593169e7"

APRILTAG_REPO = "https://github.com/Social-Evolution-and-Behavior/apriltag.git"
APRILTAG_COMMIT = "c43a9b6e6b7dcfe0e7647a78eff6655a1d743c2c"

SAM3_URL = "git+https://github.com/facebookresearch/sam3.git"
SAM3_REF = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
SAM3_ENV = "hydra-sam3"
SAM3_PYTHON = "3.12"

SLEAP_ENV = "sleap"
SLEAP_PYTHON = "3.13"
#: sleap 1.6.2 pulls sleap-nn 0.1.3; newer sleap-nn breaks the shared-memory
#: transport to the SLEAP service (AttributeError in sleap_nn.data.utils).
SLEAP_PIN = "1.6.2"

DEFAULT_PYTHON = "3.13"
MIN_TARGET_PYTHON = (3, 11)

DEFAULT_ENV_NAMES = {"cpu": "hydra", "mps": "hydra-mps", "cuda": "hydra-cuda"}

#: Conda supplies only what pip cannot: the interpreter, the ffmpeg binary and
#: the toolchain for the AprilTag build. Must match environment*.yml (tested).
CONDA_PACKAGES = ["pip", "ffmpeg", "git", "cmake", "c-compiler", "cxx-compiler"]
#: conda-forge's compiler metapackages on Windows only activate an existing
#: MSVC install; MSVC itself comes from Visual Studio Build Tools.
CONDA_PACKAGES_WINDOWS = ["pip", "ffmpeg", "git", "cmake"]

#: Package families that must never coexist in one env. Everything not in the
#: wanted set for the tier is removed before installing.
CONFLICT_FAMILIES = {
    "onnxruntime": ["onnxruntime", "onnxruntime-gpu"],
    "opencv": [
        "opencv-python",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    ],
    "tensorrt": [
        "tensorrt",
        "tensorrt-cu12",
        "tensorrt-cu12-bindings",
        "tensorrt-cu12-libs",
        "tensorrt-cu13",
        "tensorrt-cu13-bindings",
        "tensorrt-cu13-libs",
    ],
}

#: ultralytics requires ``opencv-python`` (GUI build); this package uses
#: ``opencv-python-headless``. Both write ``cv2/`` -- whichever installs last
#: wins, file by file. The override removes the GUI build from resolution.
UV_OVERRIDES = ["opencv-python; sys_platform == 'never'"]

REPO_ROOT = Path(__file__).resolve().parent


class InstallError(RuntimeError):
    """A refusal or failure with an actionable message for the user."""


# ---------------------------------------------------------------------------
# Host detection (pure functions + injectable runner for tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Host:
    system: str  # "Linux" | "Darwin" | "Windows"
    machine: str  # "x86_64" | "arm64" | "AMD64" | "aarch64"
    driver_version: Optional[str] = None  # NVIDIA driver, e.g. "570.133.20"

    @property
    def is_windows(self) -> bool:
        return self.system == "Windows"

    @property
    def is_apple_silicon(self) -> bool:
        return self.system == "Darwin" and self.machine in ("arm64", "aarch64")

    @property
    def driver_major(self) -> Optional[int]:
        return parse_driver_major(self.driver_version)


Runner = Callable[[Sequence[str]], Tuple[int, str]]


def _default_runner(argv: Sequence[str]) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return 127, ""
    return proc.returncode, proc.stdout


def parse_driver_major(text: Optional[str]) -> Optional[int]:
    """First integer of the first non-empty line, e.g. ``"570.133.20"`` -> 570."""
    if not text:
        return None
    for line in text.strip().splitlines():
        match = re.match(r"\s*(\d+)(?:\.\d+)*", line)
        if match:
            return int(match.group(1))
    return None


def detect_host(runner: Runner = _default_runner) -> Host:
    driver = None
    if shutil.which("nvidia-smi"):
        code, out = runner(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
        if code == 0 and parse_driver_major(out) is not None:
            driver = out.strip().splitlines()[0].strip()
    return Host(platform.system(), platform.machine(), driver)


def resolve_tier(requested: str, cuda: str, host: Host) -> str:
    """Return ``cpu`` | ``mps`` | ``cuda``."""
    if requested not in ("auto", "cpu", "mps", "cuda"):
        raise InstallError(f"unknown tier {requested!r}")
    if cuda == "none" and requested in ("auto", "cuda"):
        if requested == "cuda":
            raise InstallError("--tier cuda contradicts --cuda none")
        return "mps" if host.is_apple_silicon else "cpu"
    if requested == "mps" and not host.is_apple_silicon:
        raise InstallError("the mps tier needs an Apple Silicon Mac")
    if requested == "cuda" and host.system == "Darwin":
        raise InstallError("CUDA is not available on macOS")
    if requested != "auto":
        return requested
    if host.is_apple_silicon:
        return "mps"
    if host.driver_major is not None or cuda in ("12", "13"):
        return "cuda"
    return "cpu"


def resolve_cuda_major(requested: str, host: Host) -> Tuple[int, List[str]]:
    """Map the NVIDIA driver to a CUDA major (>=580 -> 13, else 12).

    An explicit ``12``/``13`` always wins, with a warning when it exceeds what
    the driver supports (that build imports fine but cannot see the GPU).
    """
    warnings: List[str] = []
    driver = host.driver_major
    if requested in ("12", "13"):
        major = int(requested)
        if major == 13 and driver is not None and driver < CUDA13_MIN_DRIVER:
            warnings.append(
                f"--cuda 13 on driver {host.driver_version} (< {CUDA13_MIN_DRIVER}): "
                "torch will import but torch.cuda.is_available() will be False. "
                "Use --cuda 12 or upgrade the driver."
            )
        return major, warnings
    if requested != "auto":
        raise InstallError(f"--cuda must be auto, 12, 13 or none (got {requested!r})")
    if driver is None:
        raise InstallError(
            "No NVIDIA driver detected (nvidia-smi missing or failing). "
            "Pass --cuda 12 or --cuda 13 to choose explicitly, or --cuda none for CPU."
        )
    return (13 if driver >= CUDA13_MIN_DRIVER else 12), warnings


# ---------------------------------------------------------------------------
# Requirement selection (pure)
# ---------------------------------------------------------------------------


def tier_key(tier: str, cuda_major: Optional[int]) -> str:
    return f"cuda{cuda_major}" if tier == "cuda" else tier


def extras_for(
    tier: str, cuda_major: Optional[int], dev: bool, docs: bool
) -> List[str]:
    extras = [tier_key(tier, cuda_major), "sam"]
    if dev:
        extras.append("dev")
    if docs:
        extras.append("docs")
    return extras


def torch_requirements(
    tier: str, cuda_major: Optional[int], host: Host
) -> Tuple[List[str], Optional[str]]:
    """Return (requirements, index_url). ``index_url`` None means PyPI."""
    if tier == "cuda":
        tag = {12: "cu128", 13: "cu130"}[int(cuda_major or 0)]
        return (
            [
                f"torch=={TORCH_VERSION}+{tag}",
                f"torchvision=={TORCHVISION_VERSION}+{tag}",
            ],
            f"{PYTORCH_INDEX}/{tag}",
        )
    reqs = [f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}"]
    if host.system == "Darwin":
        return reqs, None  # macOS wheels on PyPI are CPU/MPS builds
    # PyPI's Linux torch is a CUDA build; Windows' PyPI torch is CPU, but the
    # explicit index keeps both identical and small.
    return reqs, f"{PYTORCH_INDEX}/cpu"


def conflict_removals(tier: str, cuda_major: Optional[int]) -> List[str]:
    keep = set()
    if tier == "cuda":
        keep.add("onnxruntime-gpu")
        keep.update(
            {
                f"tensorrt-cu{cuda_major}",
                f"tensorrt-cu{cuda_major}-bindings",
                f"tensorrt-cu{cuda_major}-libs",
            }
        )
    else:
        keep.add("onnxruntime")
    removals: List[str] = []
    for names in CONFLICT_FAMILIES.values():
        removals.extend(n for n in names if n not in keep)
    return removals


def source_spec(source: str, extras: Sequence[str]) -> List[str]:
    """pip arguments installing hydra-suite from ``source`` with ``extras``."""
    extra = f"[{','.join(extras)}]" if extras else ""
    path = Path(source)
    if path.is_dir():
        return ["-e", f"{path}{extra}"]
    if source.endswith(".whl") or source.endswith(".tar.gz"):
        return [f"hydra-suite{extra} @ {Path(source).resolve().as_uri()}"]
    if "://" in source or source.startswith("git+"):
        return [f"hydra-suite{extra} @ {source}"]
    raise InstallError(f"--source {source!r} is not a directory, archive, or URL")


def source_from_direct_url(direct_url: Optional[dict]) -> Tuple[str, bool]:
    """Reproduce how hydra-suite was installed: (pip spec, editable).

    Used to install the *same code* into a sidecar env. The sidecar speaks a
    line protocol with the main process, so it must never drift to another
    version -- in particular it is never resolved from PyPI.
    """
    if not direct_url or "url" not in direct_url:
        raise InstallError(
            "Cannot tell where hydra-suite was installed from (no direct_url.json). "
            "Re-run with --source <checkout dir | git URL | wheel>."
        )
    url = direct_url["url"]
    if direct_url.get("dir_info", {}).get("editable"):
        if not url.startswith("file://"):
            raise InstallError(f"editable install from non-file URL {url!r}")
        from urllib.parse import unquote, urlparse

        parsed = urlparse(url)
        local = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:", local):  # file:///C:/... on Windows
            local = local[1:]
        return local, True
    vcs = direct_url.get("vcs_info")
    if vcs:
        return f"hydra-suite @ {vcs.get('vcs', 'git')}+{url}@{vcs['commit_id']}", False
    return f"hydra-suite @ {url}", False


# ---------------------------------------------------------------------------
# Environment layout
# ---------------------------------------------------------------------------


def env_python(prefix: Path, host: Host, kind: str) -> Path:
    if host.is_windows:
        return prefix / ("Scripts" if kind == "venv" else "") / "python.exe"
    return prefix / "bin" / "python"


def find_conda(env_var: Dict[str, str] = os.environ) -> Optional[str]:
    """Locate conda/mamba. Prefers CONDA_EXE (a real executable on Windows)."""
    exe = env_var.get("CONDA_EXE")
    if exe and Path(exe).exists():
        return exe
    for name in ("mamba", "conda"):
        found = shutil.which(name)
        if found:
            return found
    return None


def conda_env_prefix(conda: str, name: str) -> Tuple[Path, bool]:
    """Return (prefix, exists) for the named env."""
    out = subprocess.run(
        [conda, "env", "list", "--json"], capture_output=True, text=True, check=True
    ).stdout
    envs = [Path(p) for p in json.loads(out).get("envs", [])]
    for prefix in envs:
        if prefix.name == name:
            return prefix, True
    base_out = subprocess.run(
        [conda, "info", "--base"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return Path(base_out) / "envs" / name, False


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass
class Step:
    name: str
    describe: str
    argv: Optional[List[str]] = None
    func: Optional[Callable[[], None]] = None
    env: Optional[Dict[str, str]] = None
    #: Failing steps abort the install unless this is set.
    optional: bool = False

    def render(self) -> str:
        if self.argv is not None:
            return f"[{self.name}] {self.describe}\n    $ {_quote(self.argv)}"
        return f"[{self.name}] {self.describe}"


def _quote(argv: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(argv))
    import shlex

    return " ".join(shlex.quote(a) for a in argv)


@dataclass
class Options:
    tier: str = "auto"
    cuda: str = "auto"
    target: str = "auto"  # auto | conda | venv | current
    env: Optional[str] = None
    python: str = DEFAULT_PYTHON
    source: Optional[str] = None
    update: bool = False
    recreate: bool = False
    create_only: bool = False
    with_sam3_train: bool = False
    with_sleap: bool = False
    dev: bool = False
    docs: bool = False
    skip_apriltag: bool = False
    skip_doctor: bool = False
    only: Optional[str] = None
    dry_run: bool = False
    use_uv: bool = True


@dataclass
class Context:
    """Everything the plan needs, resolved up front."""

    host: Host
    tier: str
    cuda_major: Optional[int]
    target: str  # conda | venv | current
    prefix: Optional[Path]  # env prefix (None for current)
    python: Path  # interpreter to install into
    env_exists: bool
    conda: Optional[str]
    source: str
    warnings: List[str] = field(default_factory=list)


def pip_cmd(ctx: Context, opts: Options, *args: str) -> List[str]:
    """``uv pip <args>`` targeting the env, or ``python -m pip <args>``."""
    if opts.use_uv:
        return [
            str(ctx.python),
            "-m",
            "uv",
            "pip",
            args[0],
            "--python",
            str(ctx.python),
            *args[1:],
        ]
    if args[0] == "uninstall":
        return [str(ctx.python), "-m", "pip", "uninstall", "-y", *args[1:]]
    return [str(ctx.python), "-m", "pip", *args]


def build_plan(ctx: Context, opts: Options) -> List[Step]:
    steps: List[Step] = []
    py = str(ctx.python)
    host = ctx.host

    # 1. Environment
    if ctx.target == "conda":
        pkgs = CONDA_PACKAGES_WINDOWS if host.is_windows else CONDA_PACKAGES
        if ctx.env_exists and opts.recreate:
            steps.append(
                Step(
                    "env",
                    f"remove conda env {ctx.prefix.name} (--recreate)",
                    [ctx.conda, "env", "remove", "-y", "-p", str(ctx.prefix)],
                )
            )
        if ctx.env_exists and not opts.recreate:
            steps.append(
                Step(
                    "env",
                    f"update conda env {ctx.prefix.name}",
                    [
                        ctx.conda,
                        "install",
                        "-y",
                        "-p",
                        str(ctx.prefix),
                        "-c",
                        "conda-forge",
                        f"python={opts.python}",
                        *pkgs,
                    ],
                )
            )
        else:
            steps.append(
                Step(
                    "env",
                    f"create conda env {ctx.prefix.name}",
                    [
                        ctx.conda,
                        "create",
                        "-y",
                        "-p",
                        str(ctx.prefix),
                        "-c",
                        "conda-forge",
                        f"python={opts.python}",
                        *pkgs,
                    ],
                )
            )
    elif ctx.target == "venv" and not ctx.env_exists:
        steps.append(
            Step(
                "env",
                f"create venv {ctx.prefix}",
                [sys.executable, "-m", "venv", str(ctx.prefix)],
            )
        )
    if opts.create_only:
        return _filter_only(steps, opts)

    # 2. Installer tooling
    if opts.use_uv:
        steps.append(
            Step(
                "uv",
                "install uv into the env",
                [py, "-m", "pip", "install", "--quiet", "--upgrade", "pip", "uv"],
            )
        )

    # 3. Remove package families that conflict with this tier
    steps.append(
        Step(
            "conflicts",
            "remove conflicting onnxruntime / opencv / tensorrt variants",
            pip_cmd(
                ctx, opts, "uninstall", *conflict_removals(ctx.tier, ctx.cuda_major)
            ),
            optional=True,  # uninstalling absent packages is not an error
        )
    )

    # 4. torch first, from the index that matches the tier
    reqs, index = torch_requirements(ctx.tier, ctx.cuda_major, host)
    torch_args = ["install", *reqs]
    if index:
        torch_args += ["--index-url", index]
    steps.append(
        Step(
            "torch",
            f"install torch {TORCH_VERSION} for {tier_key(ctx.tier, ctx.cuda_major)}",
            pip_cmd(ctx, opts, *torch_args),
        )
    )

    # 5. hydra-suite + extras (pyproject.toml is the single source of truth)
    extras = extras_for(ctx.tier, ctx.cuda_major, opts.dev, opts.docs)
    main_args = ["install", *source_spec(ctx.source, extras)]
    if opts.use_uv:
        main_args += ["--overrides", _overrides_file()]
        # Keep the torch build chosen in step 4: never let the resolver swap it.
        main_args += ["--constraints", _torch_constraints_file(reqs)]
        if index:
            main_args += [
                "--extra-index-url",
                index,
                "--index-strategy",
                "unsafe-best-match",
            ]
    else:
        main_args += ["-c", _torch_constraints_file(reqs)]
        if index:
            main_args += ["--extra-index-url", index]
    if opts.update:
        main_args.append("--upgrade")
    steps.append(
        Step(
            "hydra",
            f"install hydra-suite[{','.join(extras)}]",
            pip_cmd(ctx, opts, *main_args),
        )
    )
    if not opts.use_uv:
        steps.append(
            Step(
                "opencv",
                "drop the GUI opencv build ultralytics pulls in",
                [py, "-m", "pip", "uninstall", "-y", "opencv-python"],
                optional=True,
            )
        )
        steps.append(
            Step(
                "opencv-headless",
                "restore opencv-python-headless",
                [
                    py,
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    "--no-deps",
                    "opencv-python-headless",
                ],
            )
        )

    # 6. CLIP (SAM3 text encoder) -- git-only, so not expressible in pyproject
    steps.append(
        Step(
            "clip",
            "install CLIP for SAM3 semantic escalation",
            pip_cmd(ctx, opts, "install", "--no-deps", f"{CLIP_URL}@{CLIP_REF}"),
            func=_require_git,
        )
    )

    # 7. AprilTag fork (C extension built from source)
    if not opts.skip_apriltag:
        steps.append(
            Step(
                "apriltag",
                f"build AprilTag fork {APRILTAG_COMMIT[:7]}",
                func=lambda: build_apriltag(ctx),
            )
        )

    # 8. Platform fixups
    if ctx.tier == "mps":
        steps.append(
            Step(
                "libomp",
                "de-duplicate libomp (torch vs other OpenMP copies)",
                [py, "-m", "hydra_suite.runtime.macos_libomp"],
            )
        )

    # 9. Optional sidecars
    if opts.with_sam3_train:
        steps.extend(sam3_train_steps(ctx, opts))
    if opts.with_sleap:
        steps.extend(sleap_steps(ctx, opts))

    # 10. Verify
    if not opts.skip_doctor:
        doctor = [py, "-m", "hydra_suite.runtime.doctor", "--tier", ctx.tier]
        if opts.with_sam3_train:
            doctor.append("--require-sam3-train")
        steps.append(Step("doctor", "verify the install (hydra doctor)", doctor))
    return _filter_only(steps, opts)


def _filter_only(steps: List[Step], opts: Options) -> List[Step]:
    if not opts.only:
        return steps
    wanted = {s.strip() for s in opts.only.split(",")}
    return [s for s in steps if s.name in wanted]


def _write_temp(name: str, lines: Sequence[str]) -> str:
    path = Path(tempfile.gettempdir()) / f"hydra-install-{os.getpid()}-{name}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _overrides_file() -> str:
    return _write_temp("overrides.txt", UV_OVERRIDES)


def _torch_constraints_file(reqs: Sequence[str]) -> str:
    return _write_temp("torch-constraints.txt", reqs)


def _require_git() -> None:
    if not shutil.which("git"):
        raise InstallError(
            "git is required to install CLIP and the AprilTag fork. Install git "
            "(conda install -c conda-forge git, or https://git-scm.com) and re-run."
        )


# ---------------------------------------------------------------------------
# AprilTag
# ---------------------------------------------------------------------------


def build_apriltag(ctx: Context) -> None:
    _require_git()
    if not shutil.which("cmake"):
        raise InstallError(
            "cmake is required to build the AprilTag fork. "
            + _compiler_hint(ctx.host)
            + " Or re-run with --skip-apriltag (AprilTag identity will be unavailable)."
        )
    prefix = _python_prefix(ctx.python)
    with tempfile.TemporaryDirectory(prefix="hydra-apriltag-") as tmp:
        src = Path(tmp) / "apriltag"
        _run(
            [shutil.which("git") or "git", "clone", "--quiet", APRILTAG_REPO, str(src)]
        )
        _run(
            ["git", "-C", str(src), "checkout", "--quiet", "--detach", APRILTAG_COMMIT]
        )
        configure = [
            "cmake",
            "-S",
            str(src),
            "-B",
            str(src / "build"),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            f"-DPython3_EXECUTABLE={ctx.python}",
            f"-DPython3_ROOT_DIR={prefix}",
            "-DPython3_FIND_STRATEGY=LOCATION",
            "-DBUILD_EXAMPLES=OFF",
        ]
        if ctx.host.is_windows:
            # Extension modules do not search PATH for DLLs (Python >= 3.8), so a
            # shared apriltag.dll in <prefix>/bin would never load. Link statically.
            configure.append("-DBUILD_SHARED_LIBS=OFF")
        try:
            _run(configure)
            _run(
                [
                    "cmake",
                    "--build",
                    str(src / "build"),
                    "--config",
                    "Release",
                    "--parallel",
                ]
            )
            _run(["cmake", "--install", str(src / "build"), "--config", "Release"])
        except subprocess.CalledProcessError as exc:
            raise InstallError(
                f"AprilTag build failed ({exc}). {_compiler_hint(ctx.host)} "
                "Or re-run with --skip-apriltag (AprilTag identity will be unavailable)."
            ) from exc
    _run([str(ctx.python), "-c", "import apriltag"])


def _compiler_hint(host: Host) -> str:
    if host.is_windows:
        return (
            "On Windows install 'Visual Studio Build Tools' with the 'Desktop "
            "development with C++' workload, plus cmake (conda install cmake)."
        )
    if host.system == "Darwin":
        return "On macOS run 'xcode-select --install' and accept the licence ('sudo xcodebuild -license')."
    return "On Linux install a C compiler and cmake (conda install -c conda-forge c-compiler cmake)."


def _python_prefix(python: Path) -> str:
    return subprocess.run(
        [str(python), "-c", "import sys; print(sys.prefix)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Sidecars
# ---------------------------------------------------------------------------


def sam3_train_steps(ctx: Context, opts: Options) -> List[Step]:
    if ctx.tier != "cuda":
        raise InstallError(
            "SAM3 training requires an NVIDIA GPU (CUDA, compute capability >= 8.0 "
            "with bf16). This host resolved to the "
            f"{ctx.tier!r} tier, so --with-sam3-train is refused. SAM3 *inference* "
            "is installed on every tier and needs nothing extra."
        )
    if not ctx.conda:
        raise InstallError(
            "SAM3 training runs in a conda sidecar env; conda/mamba was not found on PATH."
        )
    conda = ctx.conda
    side = [conda, "run", "-n", SAM3_ENV, "--no-capture-output", "python", "-m", "pip"]
    reqs, index = torch_requirements("cuda", ctx.cuda_major, ctx.host)
    return [
        Step(
            "sam3-train",
            f"create sidecar env {SAM3_ENV} (python {SAM3_PYTHON}, numpy<2)",
            [
                conda,
                "create",
                "-y",
                "-n",
                SAM3_ENV,
                "-c",
                "conda-forge",
                f"python={SAM3_PYTHON}",
                "numpy<2",
                "pip",
            ],
            func=lambda: _skip_if_env_exists(conda, SAM3_ENV),
        ),
        Step(
            "sam3-train",
            "install CUDA torch into the sidecar",
            [*side, "install", *reqs, "--index-url", str(index)],
        ),
        # sam3/model_builder.py needs pkg_resources (setuptools<81). scipy>=1.14 and
        # opencv-python-headless>=4.12 require numpy>=2; install every pin in ONE
        # resolver call so nothing drags numpy 2 back in transitively.
        Step(
            "sam3-train",
            "install sam3 training dependencies",
            [
                *side,
                "install",
                "setuptools<81",
                "numpy<2",
                "einops",
                "torchmetrics",
                "scipy<1.14",
                "decord",
                "iopath>=0.1.10",
                "opencv-python-headless<4.12",
                "pillow",
                "platformdirs",
                "pandas",
                "numba",
                "pycocotools",
                "psutil",
            ],
        ),
        Step(
            "sam3-train",
            f"install sam3 @ {SAM3_REF[:7]}",
            [*side, "install", "--no-deps", f"{SAM3_URL}@{SAM3_REF}"],
        ),
        Step(
            "sam3-train",
            "install the SAME hydra-suite source into the sidecar (--no-deps)",
            func=lambda: _install_same_hydra_into(ctx, side),
        ),
    ]


def _skip_if_env_exists(conda: str, name: str) -> None:
    _, exists = conda_env_prefix(conda, name)
    if exists:
        print(
            f"    conda env {name!r} exists; reusing it (remove it for a clean rebuild)."
        )
        raise _SkipArgv()


class _SkipArgv(Exception):
    """Raised by a step's pre-check to skip its argv."""


def main_env_direct_url(python: Path) -> Optional[dict]:
    code = (
        "import importlib.metadata as m, json\n"
        "try:\n"
        "    t = m.distribution('hydra-suite').read_text('direct_url.json')\n"
        "except m.PackageNotFoundError:\n"
        "    t = None\n"
        "print(t or 'null')\n"
    )
    out = subprocess.run(
        [str(python), "-c", code], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def _install_same_hydra_into(ctx: Context, side_pip: List[str]) -> None:
    spec, editable = source_from_direct_url(main_env_direct_url(ctx.python))
    args = ["install", "--no-deps"] + (["-e", spec] if editable else [spec])
    _run([*side_pip, *args])
    print(f"    sidecar hydra-suite <- {'editable ' if editable else ''}{spec}")


def sleap_steps(ctx: Context, opts: Options) -> List[Step]:
    if not ctx.conda:
        raise InstallError(
            "The SLEAP sidecar is a conda env; conda/mamba was not found on PATH."
        )
    conda = ctx.conda
    side = [conda, "run", "-n", SLEAP_ENV, "--no-capture-output", "python", "-m", "pip"]
    extra = "nn,nn-export-gpu" if ctx.tier == "cuda" else "nn,nn-export"
    steps = [
        Step(
            "sleap",
            f"create sidecar env {SLEAP_ENV} (python {SLEAP_PYTHON})",
            [
                conda,
                "create",
                "-y",
                "-n",
                SLEAP_ENV,
                "-c",
                "conda-forge",
                f"python={SLEAP_PYTHON}",
                "pip",
            ],
            func=lambda: _skip_if_env_exists(conda, SLEAP_ENV),
        ),
        Step(
            "sleap",
            f"install sleap[{extra}]=={SLEAP_PIN}",
            [*side, "install", f"sleap[{extra}]=={SLEAP_PIN}"],
        ),
    ]
    reqs, index = torch_requirements(ctx.tier, ctx.cuda_major, ctx.host)
    torch_args = [*side, "install", "--force-reinstall", "--no-deps", *reqs]
    if index:
        torch_args += ["--index-url", index]
    steps.append(
        Step("sleap", "pin the sidecar's torch to this host's build", torch_args)
    )
    return steps


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _run(argv: Sequence[str], env: Optional[Dict[str, str]] = None) -> None:
    print(f"    $ {_quote(argv)}", flush=True)
    subprocess.run(list(argv), check=True, env=env)


def execute(steps: Sequence[Step]) -> None:
    for i, step in enumerate(steps, 1):
        print(f"\n==> [{i}/{len(steps)}] {step.describe}", flush=True)
        try:
            if step.func is not None:
                step.func()
            if step.argv is not None:
                _run(step.argv, env={**os.environ, **(step.env or {})})
        except _SkipArgv:
            continue
        except subprocess.CalledProcessError as exc:
            if step.optional:
                print(f"    (optional step failed: {exc}; continuing)")
                continue
            raise InstallError(f"step '{step.name}' failed: {exc}") from exc


def resolve_context(opts: Options, host: Optional[Host] = None) -> Context:
    host = host or detect_host()
    tier = resolve_tier(opts.tier, opts.cuda, host)
    warnings: List[str] = []
    cuda_major = None
    if tier == "cuda":
        cuda_major, warnings = resolve_cuda_major(opts.cuda, host)

    conda = find_conda()
    target = opts.target
    if target == "auto":
        target = "conda" if conda else "venv"
    source = opts.source or str(REPO_ROOT)

    if target == "current":
        python = Path(sys.executable)
        if sys.version_info < MIN_TARGET_PYTHON:
            raise InstallError(
                f"--target current needs Python >= {'.'.join(map(str, MIN_TARGET_PYTHON))}; "
                f"this is {platform.python_version()}. Use --target conda or venv."
            )
        return Context(
            host, tier, cuda_major, target, None, python, True, conda, source, warnings
        )

    name = opts.env or DEFAULT_ENV_NAMES[tier]
    if target == "conda":
        if not conda:
            raise InstallError(
                "--target conda but conda/mamba is not on PATH. Use --target venv."
            )
        prefix, exists = conda_env_prefix(conda, name)
    else:
        prefix = (
            Path(opts.env)
            if opts.env and ("/" in opts.env or "\\" in opts.env)
            else REPO_ROOT / f".venv-{name}"
        )
        exists = env_python(prefix, host, "venv").exists()
        if sys.version_info < MIN_TARGET_PYTHON:
            raise InstallError(
                f"--target venv needs this script to run on Python >= "
                f"{'.'.join(map(str, MIN_TARGET_PYTHON))} (it becomes the venv's Python)."
            )
    python = env_python(prefix, host, target)
    return Context(
        host, tier, cuda_major, target, prefix, python, exists, conda, source, warnings
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> Options:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tier", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument(
        "--cuda",
        default=os.environ.get("CUDA_MAJOR") or "auto",
        choices=["auto", "12", "13", "none"],
        help="CUDA major (default: from the NVIDIA driver; >=580 -> 13, else 12)",
    )
    p.add_argument(
        "--target", default="auto", choices=["auto", "conda", "venv", "current"]
    )
    p.add_argument("--env", help="conda env name (or venv path)")
    p.add_argument(
        "--python", default=DEFAULT_PYTHON, help="Python version for a new conda env"
    )
    p.add_argument(
        "--source",
        help="hydra-suite source: checkout dir, git URL, or wheel (default: this checkout)",
    )
    p.add_argument(
        "--update", action="store_true", help="upgrade packages in an existing env"
    )
    p.add_argument(
        "--recreate",
        action="store_true",
        help="delete and rebuild the conda env (use for envs built before the pyproject-only layout)",
    )
    p.add_argument("--create-only", action="store_true", help="only create the env")
    p.add_argument(
        "--with-sam3-train",
        action="store_true",
        help="also build the SAM3 training sidecar (CUDA only)",
    )
    p.add_argument(
        "--with-sleap", action="store_true", help="also build the SLEAP sidecar env"
    )
    p.add_argument("--dev", action="store_true", help="include dev tools")
    p.add_argument("--docs", action="store_true", help="include docs tools")
    p.add_argument("--skip-apriltag", action="store_true")
    p.add_argument("--skip-doctor", action="store_true")
    p.add_argument(
        "--only", help="run only these comma-separated steps (e.g. clip,apriltag)"
    )
    p.add_argument(
        "--no-uv", dest="use_uv", action="store_false", help="use pip instead of uv"
    )
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ns = p.parse_args(argv)
    return Options(**{k: v for k, v in vars(ns).items()})


def main(argv: Optional[Sequence[str]] = None) -> int:
    opts = parse_args(argv)
    try:
        ctx = resolve_context(opts)
        plan = build_plan(ctx, opts)
    except InstallError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"HYDRA install: tier={tier_key(ctx.tier, ctx.cuda_major)} target={ctx.target} "
        f"python={ctx.python} source={ctx.source}"
    )
    if ctx.host.driver_version:
        print(f"  NVIDIA driver {ctx.host.driver_version}")
    for w in ctx.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if opts.dry_run:
        for step in plan:
            print(step.render())
        return 0
    try:
        execute(plan)
    except InstallError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    if ctx.target == "conda" and not opts.create_only:
        print(f"\nDone. Activate with:  conda activate {ctx.prefix.name}")
    elif ctx.target == "venv":
        act = ctx.prefix / (
            "Scripts\\activate" if ctx.host.is_windows else "bin/activate"
        )
        print(f"\nDone. Activate with:  {act}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
