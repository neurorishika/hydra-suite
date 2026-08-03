"""SAM2 checkpoint catalog + HF-managed download (mirrors vitpose_checkpoints)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from hydra_suite.paths import get_models_dir


@dataclass(frozen=True)
class Sam2Entry:
    repo_id: str
    filename: str
    config_name: str  # sam2 package config (e.g. "configs/sam2.1/sam2.1_hiera_b+.yaml")


# NOTE: repo_id/filename/config_name pinned to the `sam2` package's published
# assets at implementation time (verify against the installed sam2 version).
SAM2_VARIANTS: dict[str, Sam2Entry] = {
    "sam2.1-hiera-tiny": Sam2Entry(
        "facebook/sam2.1-hiera-tiny",
        "sam2.1_hiera_tiny.pt",
        "configs/sam2.1/sam2.1_hiera_t.yaml",
    ),
    "sam2.1-hiera-small": Sam2Entry(
        "facebook/sam2.1-hiera-small",
        "sam2.1_hiera_small.pt",
        "configs/sam2.1/sam2.1_hiera_s.yaml",
    ),
    "sam2.1-hiera-base_plus": Sam2Entry(
        "facebook/sam2.1-hiera-base-plus",
        "sam2.1_hiera_base_plus.pt",
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
    ),
    "sam2.1-hiera-large": Sam2Entry(
        "facebook/sam2.1-hiera-large",
        "sam2.1_hiera_large.pt",
        "configs/sam2.1/sam2.1_hiera_l.yaml",
    ),
}

DEFAULT_VARIANT = "sam2.1-hiera-base_plus"


def available_variants() -> list[str]:
    return list(SAM2_VARIANTS.keys())


def _cache_dir(cache_dir: Path | None) -> Path:
    return Path(cache_dir) if cache_dir is not None else Path(get_models_dir()) / "sam2"


def ensure_checkpoint(
    variant: str, *, allow_download: bool = True, cache_dir: Path | None = None
) -> Path:
    """Return the cached SAM2 checkpoint path, downloading from HF if needed."""
    if variant not in SAM2_VARIANTS:
        raise ValueError(
            f"Unknown SAM2 variant {variant!r}. "
            f"Available: {', '.join(available_variants())}."
        )
    entry = SAM2_VARIANTS[variant]
    cdir = _cache_dir(cache_dir)
    dest = cdir / f"{variant}.pt"
    if dest.exists():
        return dest
    if not allow_download:
        raise ValueError(
            f"SAM2 variant {variant!r} is not downloaded and downloads are "
            f"disabled (offline). Download it once with network access."
        )
    cdir.mkdir(parents=True, exist_ok=True)
    src = Path(hf_hub_download(repo_id=entry.repo_id, filename=entry.filename))
    dest.write_bytes(src.read_bytes())
    return dest
