"""SAM3 checkpoint catalog + a probe that never triggers a download.

Two facts drive this module. ``sam3.pt`` is 3.45 GB and is NOT in
ultralytics' ``GITHUB_ASSETS_NAMES``, so it comes from the public
``facebook/sam3`` HF repo. And ultralytics AutoUpdate pip-installs ``clip``
and ``ftfy`` on first use -- unacceptable on an offline or shared install.
``probe_availability`` therefore checks the Python deps and the on-disk
checkpoint BEFORE anything can reach ultralytics, and the GUI uses it to
disable the action with a reason instead of failing at click time.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from hydra_suite.paths import get_models_dir


@dataclass(frozen=True)
class Sam3Entry:
    repo_id: str
    filename: str


# Pinned repo + filename, same discipline as SAM2_VARIANTS. Verify against
# the published `facebook/sam3` assets if this ever fails to download.
SAM3_VARIANTS: dict[str, Sam3Entry] = {
    "sam3": Sam3Entry("facebook/sam3", "sam3.pt"),
}

DEFAULT_VARIANT = "sam3"

# Imports ultralytics AutoUpdate would otherwise install behind our back.
REQUIRED_PACKAGES = ("ultralytics", "clip", "ftfy")


def available_variants() -> list[str]:
    return list(SAM3_VARIANTS.keys())


def _find_spec(name: str):  # seam for tests
    return importlib.util.find_spec(name)


def _has_predictor_symbol() -> bool:  # seam for tests
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor  # noqa: F401
    except Exception:
        return False
    return True


def _cache_dir(cache_dir: Path | None) -> Path:
    return Path(cache_dir) if cache_dir is not None else Path(get_models_dir()) / "sam3"


def checkpoint_path(
    variant: str = DEFAULT_VARIANT, cache_dir: Path | None = None
) -> Path:
    return _cache_dir(cache_dir) / f"{variant}.pt"


def probe_availability(
    variant: str = DEFAULT_VARIANT, cache_dir: Path | None = None
) -> tuple[bool, str]:
    """(usable, reason). Never downloads, never imports ultralytics lazily."""
    if variant not in SAM3_VARIANTS:
        return False, f"Unknown SAM3 variant {variant!r}."
    for pkg in REQUIRED_PACKAGES:
        if _find_spec(pkg) is None:
            return False, (
                f"Python package {pkg!r} is missing. Install the SAM3 extra: "
                "pip install 'hydra-suite[sam3]'."
            )
    if not _has_predictor_symbol():
        return False, (
            "The installed ultralytics has no SAM3SemanticPredictor "
            "(needs >= 8.4.34)."
        )
    if not checkpoint_path(variant, cache_dir).exists():
        return False, (
            f"The SAM3 checkpoint ({variant}, ~3.45 GB) is not downloaded. "
            "Download it once from the semantic escalation dialog."
        )
    return True, ""


def ensure_checkpoint(
    variant: str = DEFAULT_VARIANT,
    *,
    allow_download: bool = True,
    cache_dir: Path | None = None,
) -> Path:
    """Return the cached SAM3 checkpoint path, downloading from HF if needed."""
    if variant not in SAM3_VARIANTS:
        raise ValueError(
            f"Unknown SAM3 variant {variant!r}. "
            f"Available: {', '.join(available_variants())}."
        )
    dest = checkpoint_path(variant, cache_dir)
    if dest.exists():
        return dest
    if not allow_download:
        raise ValueError(
            f"SAM3 variant {variant!r} is not downloaded and downloads are "
            "disabled (offline). Download it once with network access."
        )
    entry = SAM3_VARIANTS[variant]
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = Path(hf_hub_download(repo_id=entry.repo_id, filename=entry.filename))
    dest.write_bytes(src.read_bytes())
    return dest
