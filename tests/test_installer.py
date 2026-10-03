"""Unit tests for the stdlib-only installer (install.py at the repo root).

Pure logic only: platform/CUDA resolution, requirement selection, plan
building and source resolution. Nothing here creates an env or runs pip.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_installer():
    spec = importlib.util.spec_from_file_location("hydra_install", REPO / "install.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hydra_install"] = module
    spec.loader.exec_module(module)
    return module


inst = _load_installer()

LINUX = inst.Host("Linux", "x86_64")
WINDOWS = inst.Host("Windows", "AMD64")
MAC_ARM = inst.Host("Darwin", "arm64")
MAC_INTEL = inst.Host("Darwin", "x86_64")


def linux_gpu(driver: str) -> "inst.Host":
    return inst.Host("Linux", "x86_64", driver)


def windows_gpu(driver: str) -> "inst.Host":
    return inst.Host("Windows", "AMD64", driver)


# ---------------------------------------------------------------------------
# Driver -> CUDA major
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smi_output, expected",
    [
        ("570.133.20\n", 570),
        ("595.91.07\n595.91.07\n", 595),  # one line per GPU
        ("  580.65.06", 580),
        ("", None),
        ("NVIDIA-SMI has failed", None),
    ],
)
def test_parse_driver_major(smi_output, expected):
    assert inst.parse_driver_major(smi_output) == expected


@pytest.mark.parametrize(
    "driver, major",
    [
        ("525.60.13", 12),
        ("570.133.20", 12),  # diptera
        ("579.99", 12),
        ("580.65.06", 13),  # first CUDA 13 driver
        ("595.91.07", 13),  # mehek
    ],
)
def test_auto_cuda_major_follows_driver(driver, major):
    got, warnings = inst.resolve_cuda_major("auto", linux_gpu(driver))
    assert got == major
    assert warnings == []


def test_explicit_cuda_override_wins_with_warning_when_driver_too_old():
    got, warnings = inst.resolve_cuda_major("13", linux_gpu("570.133.20"))
    assert got == 13
    assert warnings and "is_available() will be False" in warnings[0]


def test_explicit_cuda_12_on_new_driver_is_silent():
    assert inst.resolve_cuda_major("12", linux_gpu("595.91.07")) == (12, [])


def test_auto_cuda_without_driver_refuses():
    with pytest.raises(inst.InstallError, match="No NVIDIA driver"):
        inst.resolve_cuda_major("auto", LINUX)


def test_detect_host_reads_nvidia_smi(monkeypatch):
    monkeypatch.setattr(inst.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(inst.platform, "system", lambda: "Linux")
    monkeypatch.setattr(inst.platform, "machine", lambda: "x86_64")
    host = inst.detect_host(lambda argv: (0, "570.133.20\n570.133.20\n"))
    assert host.driver_version == "570.133.20"
    assert host.driver_major == 570


def test_detect_host_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(inst.shutil, "which", lambda name: None)
    host = inst.detect_host(lambda argv: pytest.fail("must not run nvidia-smi"))
    assert host.driver_version is None


# ---------------------------------------------------------------------------
# Tier resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host, cuda, expected",
    [
        (MAC_ARM, "auto", "mps"),
        (MAC_INTEL, "auto", "cpu"),
        (LINUX, "auto", "cpu"),
        (WINDOWS, "auto", "cpu"),
        (linux_gpu("570.1"), "auto", "cuda"),
        (windows_gpu("595.1"), "auto", "cuda"),
        (linux_gpu("595.1"), "none", "cpu"),
        (LINUX, "12", "cuda"),  # explicit CUDA on a box whose smi is broken
    ],
)
def test_auto_tier(host, cuda, expected):
    assert inst.resolve_tier("auto", cuda, host) == expected


@pytest.mark.parametrize(
    "tier, host, message",
    [
        ("mps", LINUX, "Apple Silicon"),
        ("cuda", MAC_ARM, "not available on macOS"),
    ],
)
def test_impossible_tiers_refuse(tier, host, message):
    with pytest.raises(inst.InstallError, match=message):
        inst.resolve_tier(tier, "auto", host)


# ---------------------------------------------------------------------------
# Requirement selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tier, major, host, tag, index",
    [
        ("cuda", 12, linux_gpu("570"), "+cu128", "/cu128"),
        ("cuda", 13, linux_gpu("595"), "+cu130", "/cu130"),
        ("cuda", 12, windows_gpu("570"), "+cu128", "/cu128"),
        ("cpu", None, LINUX, "", "/cpu"),
        ("cpu", None, WINDOWS, "", "/cpu"),
        ("mps", None, MAC_ARM, "", None),
    ],
)
def test_torch_build_matches_tier(tier, major, host, tag, index):
    reqs, url = inst.torch_requirements(tier, major, host)
    assert reqs == [
        f"torch=={inst.TORCH_VERSION}{tag}",
        f"torchvision=={inst.TORCHVISION_VERSION}{tag}",
    ]
    if index is None:
        assert url is None
    else:
        assert url.endswith(index)


def test_cuda_pins_carry_the_local_tag():
    """The +cuXXX tag is what forces the PyTorch index; never drop it."""
    for major in (12, 13):
        reqs, _ = inst.torch_requirements("cuda", major, linux_gpu("595"))
        assert all(re.search(r"\+cu1\d\d$", r) for r in reqs)


def test_extras_select_exactly_one_tier_plus_sam():
    assert inst.extras_for("cuda", 12, False, False) == ["cuda12", "sam"]
    assert inst.extras_for("mps", None, True, True) == ["mps", "sam", "dev", "docs"]


def test_conflict_removals_keep_only_the_tier_family():
    cuda12 = inst.conflict_removals("cuda", 12)
    assert "onnxruntime" in cuda12 and "onnxruntime-gpu" not in cuda12
    assert "tensorrt-cu13" in cuda12 and "tensorrt-cu12" not in cuda12
    assert "opencv-python" in cuda12
    assert "opencv-python-headless" not in cuda12
    cpu = inst.conflict_removals("cpu", None)
    assert "onnxruntime-gpu" in cpu and "onnxruntime" not in cpu
    assert "tensorrt-cu12" in cpu


def test_extras_exist_in_pyproject():
    import tomllib

    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    for tier, major in (("cpu", None), ("mps", None), ("cuda", 12), ("cuda", 13)):
        for extra in inst.extras_for(tier, major, True, True):
            assert extra in extras, extra


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------


def test_source_spec_checkout_dir_is_editable(tmp_path):
    assert inst.source_spec(str(tmp_path), ["cpu", "sam"]) == [
        "-e",
        f"{tmp_path}[cpu,sam]",
    ]


def test_source_spec_git_url():
    spec = inst.source_spec("git+https://github.com/x/hydra-suite@main", ["mps", "sam"])
    assert spec == ["hydra-suite[mps,sam] @ git+https://github.com/x/hydra-suite@main"]


def test_source_spec_rejects_garbage():
    with pytest.raises(inst.InstallError):
        inst.source_spec("not-a-thing", ["cpu"])


def test_sidecar_source_reproduces_editable_install():
    spec, editable = inst.source_from_direct_url(
        {"url": "file:///home/me/hydra-suite", "dir_info": {"editable": True}}
    )
    assert (spec, editable) == ("/home/me/hydra-suite", True)


def test_sidecar_source_reproduces_windows_editable_install():
    spec, editable = inst.source_from_direct_url(
        {"url": "file:///C:/Users/me/hydra-suite", "dir_info": {"editable": True}}
    )
    assert (spec, editable) == ("C:/Users/me/hydra-suite", True)


def test_sidecar_source_pins_the_git_commit():
    spec, editable = inst.source_from_direct_url(
        {
            "url": "https://github.com/x/hydra-suite.git",
            "vcs_info": {"vcs": "git", "commit_id": "abc123"},
        }
    )
    assert spec == "hydra-suite @ git+https://github.com/x/hydra-suite.git@abc123"
    assert editable is False


def test_sidecar_source_without_direct_url_refuses():
    with pytest.raises(inst.InstallError, match="--source"):
        inst.source_from_direct_url(None)


# ---------------------------------------------------------------------------
# Plan building
# ---------------------------------------------------------------------------


def _ctx(host, tier, major=None, target="current", conda="conda"):
    prefix = None if target == "current" else Path("/envs/hydra-x")
    python = (
        Path(sys.executable)
        if target == "current"
        else inst.env_python(prefix, host, target)
    )
    return inst.Context(
        host, tier, major, target, prefix, python, False, conda, str(REPO)
    )


def _argv_text(steps):
    return "\n".join(" ".join(s.argv) for s in steps if s.argv)


def test_plan_order_puts_torch_before_hydra():
    steps = inst.build_plan(_ctx(LINUX, "cpu"), inst.Options(skip_doctor=True))
    names = [s.name for s in steps]
    assert names.index("torch") < names.index("hydra") < names.index("clip")
    assert "apriltag" in names


def test_plan_installs_cuda_torch_from_pytorch_index_only():
    steps = inst.build_plan(_ctx(linux_gpu("570"), "cuda", 12), inst.Options())
    torch_step = next(s for s in steps if s.name == "torch")
    assert "--index-url" in torch_step.argv
    assert torch_step.argv[torch_step.argv.index("--index-url") + 1].endswith("/cu128")
    assert "--extra-index-url" not in torch_step.argv


def test_plan_main_install_uses_overrides_and_torch_constraints():
    steps = inst.build_plan(_ctx(MAC_ARM, "mps"), inst.Options())
    hydra = next(s for s in steps if s.name == "hydra")
    assert "--overrides" in hydra.argv and "--constraints" in hydra.argv
    overrides = Path(hydra.argv[hydra.argv.index("--overrides") + 1]).read_text()
    assert "opencv-python;" in overrides
    assert any(a.endswith("[mps,sam]") for a in hydra.argv)


def test_plan_mps_runs_libomp_fixup_and_doctor():
    steps = inst.build_plan(_ctx(MAC_ARM, "mps"), inst.Options())
    text = _argv_text(steps)
    assert "hydra_suite.runtime.macos_libomp" in text
    assert "hydra_suite.runtime.doctor --tier mps" in text


def test_plan_without_uv_uses_pip_and_restores_headless_opencv():
    steps = inst.build_plan(_ctx(LINUX, "cpu"), inst.Options(use_uv=False))
    names = [s.name for s in steps]
    assert "uv" not in names and "opencv-headless" in names
    assert " -m uv " not in _argv_text(steps)


def test_conda_env_on_windows_omits_compiler_metapackages():
    steps = inst.build_plan(
        _ctx(windows_gpu("595"), "cuda", 13, target="conda"), inst.Options()
    )
    env = next(s for s in steps if s.name == "env")
    assert "cxx-compiler" not in env.argv and "cmake" in env.argv


def test_windows_env_python_layout():
    assert inst.env_python(Path("C:/envs/h"), WINDOWS, "conda") == Path(
        "C:/envs/h/python.exe"
    )
    assert inst.env_python(Path("C:/envs/h"), WINDOWS, "venv") == Path(
        "C:/envs/h/Scripts/python.exe"
    )
    assert inst.env_python(Path("/envs/h"), LINUX, "venv") == Path("/envs/h/bin/python")


def test_create_only_stops_after_env():
    steps = inst.build_plan(
        _ctx(LINUX, "cpu", target="conda"), inst.Options(create_only=True)
    )
    assert [s.name for s in steps] == ["env"]


def test_only_filters_steps():
    steps = inst.build_plan(_ctx(LINUX, "cpu"), inst.Options(only="clip,apriltag"))
    assert [s.name for s in steps] == ["clip", "apriltag"]


@pytest.mark.parametrize(
    "tier, host", [("cpu", LINUX), ("mps", MAC_ARM), ("cpu", WINDOWS)]
)
def test_sam3_training_refused_off_cuda(tier, host):
    with pytest.raises(inst.InstallError, match="requires an NVIDIA GPU"):
        inst.build_plan(_ctx(host, tier), inst.Options(with_sam3_train=True))


def test_sam3_training_sidecar_uses_the_resolved_cuda_build():
    steps = inst.build_plan(
        _ctx(linux_gpu("570"), "cuda", 12), inst.Options(with_sam3_train=True)
    )
    text = _argv_text([s for s in steps if s.name == "sam3-train"])
    assert f"torch=={inst.TORCH_VERSION}+cu128" in text
    assert f"sam3.git@{inst.SAM3_REF}" in text
    assert "numpy<2" in text


def test_sleap_sidecar_is_pinned():
    steps = inst.build_plan(_ctx(LINUX, "cpu"), inst.Options(with_sleap=True))
    text = _argv_text([s for s in steps if s.name == "sleap"])
    assert f"=={inst.SLEAP_PIN}" in text


# ---------------------------------------------------------------------------
# Conda files stay in sync with the installer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["environment.yml", "environment-mps.yml", "environment-cuda.yml"]
)
def test_conda_files_hold_only_non_pip_packages(name):
    lines = (REPO / name).read_text().splitlines()
    deps = [
        ln[2:].strip()
        for ln in lines
        if ln.startswith("- ") and not ln.startswith("- conda-forge")
    ]
    assert deps[0].startswith("python=")
    assert deps[1:] == inst.CONDA_PACKAGES


def test_requirements_files_are_gone():
    """pyproject.toml is the single source of truth; do not reintroduce these."""
    assert sorted(p.name for p in REPO.glob("requirements*.txt")) == []


# ---------------------------------------------------------------------------
# Tested-version pins (constraints/)
# ---------------------------------------------------------------------------


def test_default_install_uses_tested_pins():
    files = inst.pin_files(_ctx(linux_gpu("570"), "cuda", 12), inst.Options())
    assert [f.name for f in files] == ["base.txt", "cuda12.txt"]
    files = inst.pin_files(_ctx(MAC_ARM, "mps"), inst.Options())
    assert [f.name for f in files] == ["base.txt"]


def test_latest_drops_tested_pins_but_keeps_user_constraints(tmp_path):
    user = tmp_path / "lock.txt"
    user.write_text("numpy==2.3.5\n")
    files = inst.pin_files(
        _ctx(LINUX, "cpu"), inst.Options(latest=True, constraints=str(user))
    )
    assert files == [user.resolve()]


def _pin_names(path):
    names = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(
                re.split(r"[=<>;\[ ]", line, maxsplit=1)[0].lower().replace("_", "-")
            )
    return names


def test_pins_cover_every_declared_dependency():
    """A dependency added to pyproject must be re-locked (tools/lock_constraints.py)."""
    import tomllib

    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    base = _pin_names(REPO / "constraints" / "base.txt")
    unpinned_by_design = {"torch", "torchvision", "hydra-suite"}

    def names(reqs):
        for req in reqs:
            name = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req).group(1)
            name = name.lower().replace("_", "-")
            if name not in unpinned_by_design:
                yield name

    extras = project["optional-dependencies"]
    for name in names(project["dependencies"]):
        assert name in base, f"{name} declared but not pinned in constraints/base.txt"
    for extra in ("cpu", "mps", "onnx-tools", "sam", "dev", "docs"):
        for name in names(extras[extra]):
            assert name in base, f"[{extra}] {name} not pinned in constraints/base.txt"
    for major in (12, 13):
        pinned = base | _pin_names(REPO / "constraints" / f"cuda{major}.txt")
        for name in names(extras[f"cuda{major}"]):
            assert name in pinned, f"[cuda{major}] {name} not pinned"


def test_pins_never_fix_torch_or_its_cuda_wheels():
    for path in (REPO / "constraints").glob("*.txt"):
        for name in _pin_names(path):
            assert not name.startswith(("torch", "nvidia-", "triton")), (
                path.name,
                name,
            )


def test_sam3_training_on_windows_adds_triton_windows():
    steps = inst.build_plan(
        _ctx(windows_gpu("595"), "cuda", 13), inst.Options(with_sam3_train=True)
    )
    text = _argv_text([s for s in steps if s.name == "sam3-train"])
    assert "triton-windows" in text
    linux = inst.build_plan(
        _ctx(linux_gpu("595"), "cuda", 13), inst.Options(with_sam3_train=True)
    )
    assert "triton-windows" not in _argv_text(linux)


# ---------------------------------------------------------------------------
# Adversarial-review regressions
# ---------------------------------------------------------------------------


def test_only_with_unknown_step_is_an_error():
    with pytest.raises(inst.InstallError, match="aprilag"):
        inst.build_plan(_ctx(LINUX, "cpu"), inst.Options(only="aprilag"))


def test_sleap_sidecar_pins_sleap_nn_too():
    steps = inst.build_plan(_ctx(LINUX, "cpu"), inst.Options(with_sleap=True))
    assert f"sleap-nn=={inst.SLEAP_NN_PIN}" in _argv_text(
        [s for s in steps if s.name == "sleap"]
    )


def test_existing_conda_env_keeps_its_python_unless_asked():
    ctx = _ctx(LINUX, "cpu", target="conda")
    ctx.env_exists = True
    env = next(s for s in inst.build_plan(ctx, inst.Options()) if s.name == "env")
    assert not any(a.startswith("python=") for a in env.argv)
    env = next(
        s for s in inst.build_plan(ctx, inst.Options(python="3.12")) if s.name == "env"
    )
    assert "python=3.12" in env.argv


def test_new_conda_env_gets_the_default_python():
    env = next(
        s
        for s in inst.build_plan(_ctx(LINUX, "cpu", target="conda"), inst.Options())
        if s.name == "env"
    )
    assert f"python={inst.DEFAULT_PYTHON}" in env.argv


@pytest.mark.parametrize(
    "version, ok",
    [((3, 10), False), ((3, 11), True), ((3, 13), True), ((3, 14), False)],
)
def test_target_python_window(version, ok):
    if ok:
        inst._check_target_python("x", version)
    else:
        with pytest.raises(inst.InstallError, match="3.11-3.13"):
            inst._check_target_python("x", version)


def test_conda_preferred_over_mamba(monkeypatch):
    found = {"mamba": "/opt/bin/mamba", "conda": "/opt/bin/conda"}
    monkeypatch.setattr(inst.shutil, "which", found.get)
    assert inst.find_conda({}) == "/opt/bin/conda"


def test_windows_conda_bat_is_swapped_for_conda_exe(monkeypatch, tmp_path):
    bat = tmp_path / "condabin" / "conda.bat"
    exe = tmp_path / "Scripts" / "conda.exe"
    bat.parent.mkdir()
    exe.parent.mkdir()
    bat.write_text("")
    exe.write_text("")
    monkeypatch.setattr(
        inst.shutil, "which", lambda name: str(bat) if name == "conda" else None
    )
    assert inst.find_conda({}) == str(exe)


def test_mamba2_run_drops_no_capture_output(monkeypatch):
    class R:
        stdout = "2.8.0\n"

    monkeypatch.setattr(inst.subprocess, "run", lambda *a, **k: R())
    assert inst.conda_run_flags("/opt/bin/mamba") == []
    assert inst.conda_run_flags("/opt/bin/conda") == ["--no-capture-output"]


def test_env_bin_dirs_cover_windows_conda_layout():
    dirs = inst.env_bin_dirs(Path("C:/e"), WINDOWS, "conda")
    assert Path("C:/e/Library/bin") in dirs and Path("C:/e/Scripts") in dirs
    assert inst.env_bin_dirs(Path("/e"), LINUX, "conda") == [Path("/e/bin")]
