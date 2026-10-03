"""Every third-party module imported anywhere in src/ is declared in pyproject.

pyproject.toml is the single source of truth for Python dependencies. Before
this guard, optuna/umap/yaml were provided only by the conda env files, so a
pip install crashed the moment the parameter helper opened. This test scans
the source (it needs nothing installed) and fails on any import whose
distribution is absent from the core dependencies and every extra.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "hydra_suite"

#: import name -> acceptable distribution names (any one declared suffices).
IMPORT_TO_DISTS = {
    "PIL": {"pillow"},
    "PyNvVideoCodec": {"pynvvideocodec"},
    "PySide6": {"pyside6"},
    "cv2": {"opencv-python-headless"},
    "cupy": {"cupy-cuda12x", "cupy-cuda13x"},
    "cupyx": {"cupy-cuda12x", "cupy-cuda13x"},
    "huggingface_hub": {"huggingface-hub"},
    "imageio_ffmpeg": {"imageio-ffmpeg"},
    "onnxruntime": {"onnxruntime", "onnxruntime-gpu"},
    "skimage": {"scikit-image"},
    "sklearn": {"scikit-learn"},
    "tensorrt": {"tensorrt-cu12", "tensorrt-cu13"},
    "umap": {"umap-learn"},
    "yaml": {"pyyaml"},
}

#: Imported, but deliberately NOT a pyproject dependency of the main env.
EXTERNAL = {
    "apriltag": "C-extension fork built from source by install.py",
    "clip": "git-only (PEP 508 direct refs break PyPI upload); install.py pins it",
    "sam3": "lives only in the hydra-sam3 training sidecar env",
    "sleap": "lives only in the sleap sidecar env",
    "sleap_nn": "lives only in the sleap sidecar env",
}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared() -> set[str]:
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    reqs = list(project["dependencies"])
    for extra, items in project["optional-dependencies"].items():
        if extra == "sam3-train":  # sidecar-only; never in the main env
            continue
        reqs.extend(items)
    names = set()
    for req in reqs:
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
        if match and _normalize(match.group(1)) != "hydra-suite":
            names.add(_normalize(match.group(1)))
    return names


def _imported_modules() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                found.setdefault(top, set()).add(str(path.relative_to(REPO)))
    return found


def _third_party(modules: dict[str, set[str]]) -> dict[str, set[str]]:
    stdlib = set(sys.stdlib_module_names)
    return {
        m: files
        for m, files in modules.items()
        if m not in stdlib and m != "hydra_suite" and not m.startswith("_")
    }


def test_every_third_party_import_is_declared():
    declared = _declared()
    missing = {}
    for module, files in sorted(_third_party(_imported_modules()).items()):
        if module in EXTERNAL:
            continue
        candidates = IMPORT_TO_DISTS.get(module, {_normalize(module)})
        if not candidates & declared:
            missing[module] = sorted(files)[:3]
    assert not missing, (
        "Imported in src/ but not declared in pyproject.toml (core or an extra):\n"
        + "\n".join(f"  {m}  (e.g. {', '.join(f)})" for m, f in missing.items())
        + "\nDeclare it, map it in IMPORT_TO_DISTS, or justify it in EXTERNAL."
    )


def test_external_allowlist_has_no_stale_entries():
    imported = _third_party(_imported_modules())
    stale = sorted(set(EXTERNAL) - set(imported) - {"sleap"})
    assert not stale, f"EXTERNAL lists modules src/ no longer imports: {stale}"


def test_no_declared_core_dependency_is_unused():
    """Catch dead weight like the never-imported annoy/torchaudio."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    imported_dists = set()
    for module in _third_party(_imported_modules()):
        imported_dists |= IMPORT_TO_DISTS.get(module, {_normalize(module)})
    # Runtime-only deps that are never `import`ed by name.
    indirect = {"cmaes", "torchvision"}
    unused = []
    for req in project["dependencies"]:
        name = _normalize(re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req).group(1))
        if name not in imported_dists and name not in indirect:
            unused.append(name)
    assert not unused, f"core dependencies never imported by src/: {unused}"
