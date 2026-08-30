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
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

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

# Stated wherever the download is offered, so its size is never a surprise.
CHECKPOINT_SIZE_GB = 3.45

# Imports ultralytics AutoUpdate would otherwise install behind our back.
REQUIRED_PACKAGES = ("ultralytics", "clip", "ftfy")

DEFAULT_INSTALL_HINT = "pip install 'hydra-suite[sam3]'"
# `clip` is NOT in the sam3 extra and cannot be: it is a PEP 508 direct
# reference, which PyPI rejects in uploaded metadata. Pointing at the extra
# for it named an install that could never satisfy the check. Same command
# as the user guide's install section.
# ultralytics builds SAM3's text encoder with `clip.simple_tokenizer.
# SimpleTokenizer()` and then CALLS it (build_sam3.py:159). openai/CLIP's
# SimpleTokenizer has no __call__, so pointing users there yields a probe
# that says "ready" and a run that dies with an opaque TypeError deep in the
# text encoder. ultralytics' own fork adds the __call__. Verified on CUDA.
INSTALL_HINTS = {"clip": "pip install git+https://github.com/ultralytics/CLIP.git"}

# The weights live behind a licence gate, so "download it for you" is only
# true once the user has accepted it and logged in on THIS machine.
GATED_REPO_HINT = (
    "The SAM3 weights are gated by Meta. Downloading them needs two one-off "
    "steps on this machine:\n"
    "  1. Open https://huggingface.co/{repo} and accept the licence.\n"
    "  2. Authenticate: `hf auth login` (or set HF_TOKEN).\n"
    "Then start the run again."
)


class Sam3DownloadNotAuthorized(RuntimeError):
    """The SAM3 weights exist but this machine may not fetch them yet."""


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


class Sam3Availability(NamedTuple):
    """Structured probe result.

    ``usable`` alone conflates two very different states, and the GUI must
    tell them apart: a missing Python dependency is genuinely unusable and
    the action stays disabled, whereas a merely-undownloaded checkpoint is
    one confirmed download away from working -- gating the action on it
    made the feature unreachable, because the only place offering the
    download sat BEHIND the disabled button.
    """

    usable: bool
    reason: str
    checkpoint_missing: bool = False

    @property
    def actionable(self) -> bool:
        """True when the user can start a run (possibly after a download)."""
        return self.usable or self.checkpoint_missing


def _clip_tokenizer_problem() -> str:
    """Non-empty when the installed `clip` is the wrong fork for SAM3.

    ultralytics constructs the text encoder with
    ``clip.simple_tokenizer.SimpleTokenizer()`` and then CALLS it. openai/CLIP
    ships that class without ``__call__``, so an otherwise healthy install
    reports "ready", downloads 3.45 GB, loads the model, and only then dies
    with ``TypeError: 'SimpleTokenizer' object is not callable`` from inside
    the text encoder. Catch it while it is still a one-line fix.
    """
    try:
        import clip.simple_tokenizer as st
    except ImportError:
        # Not installed at all -- that is the missing-package check's job,
        # which runs first and gives the better message. Say nothing here.
        return ""
    try:
        if callable(st.SimpleTokenizer()):
            return ""
    except Exception as exc:  # a clip that cannot even build its tokenizer
        return (
            f"The installed 'clip' package is unusable for SAM3 ({exc}). "
            f"{INSTALL_HINTS['clip']}"
        )
    return (
        "The installed 'clip' package is the wrong fork: SAM3 needs a "
        "SimpleTokenizer that can be called, which openai/CLIP does not "
        f"provide. Replace it with: {INSTALL_HINTS['clip']}"
    )


def probe_availability(
    variant: str = DEFAULT_VARIANT, cache_dir: Path | None = None
) -> Sam3Availability:
    """Structured availability. Never downloads, never imports ultralytics."""
    if variant not in SAM3_VARIANTS:
        return Sam3Availability(False, f"Unknown SAM3 variant {variant!r}.")
    for pkg in REQUIRED_PACKAGES:
        if _find_spec(pkg) is None:
            return Sam3Availability(
                False,
                f"Python package {pkg!r} is missing. Install it with: "
                f"{INSTALL_HINTS.get(pkg, DEFAULT_INSTALL_HINT)}",
            )
    wrong_clip = _clip_tokenizer_problem()
    if wrong_clip:
        return Sam3Availability(False, wrong_clip)
    if not _has_predictor_symbol():
        return Sam3Availability(
            False,
            "The installed ultralytics has no SAM3SemanticPredictor "
            "(needs >= 8.4.34).",
        )
    if not checkpoint_path(variant, cache_dir).exists():
        return Sam3Availability(
            False,
            f"The SAM3 checkpoint ({variant}, ~{CHECKPOINT_SIZE_GB:.2f} GB) has "
            "not been downloaded yet. It will be downloaded once, with your "
            "confirmation, before the first run starts. The weights are "
            "licence-gated: if you have not accepted the licence at "
            "https://huggingface.co/facebook/sam3 and run `hf auth login` on "
            "this machine, the download will stop and tell you so.",
            checkpoint_missing=True,
        )
    return Sam3Availability(True, "")


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
    try:
        src = Path(hf_hub_download(repo_id=entry.repo_id, filename=entry.filename))
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        # facebook/sam3 is a GATED repo: without an accepted licence and a
        # token, hf_hub_download raises a bare 401 that says nothing about
        # what the user has to go and do. Verified on the CUDA box.
        raise Sam3DownloadNotAuthorized(
            GATED_REPO_HINT.format(repo=entry.repo_id)
        ) from exc
    # hf_hub_download returns a path inside the HF cache SNAPSHOT dir, which
    # on Linux is a symlink into ../../blobs/<sha>. Hardlinking that entry
    # copies the symlink itself, so `dest` inherits a relative target that
    # means nothing where it now lives -- a dangling link whose .exists() is
    # False, so the probe re-downloads 3.3 GB on every run, forever, and the
    # feature can never start. Resolve to the real blob first. (Reproduced on
    # the CUDA box; macOS never hit it because nothing had downloaded yet.)
    src = src.resolve()
    # NOT dest.write_bytes(src.read_bytes()): that holds all 3.45 GB in RAM at
    # once and can OOM a 16 GB laptop. Hardlink when the HF cache is on the
    # same volume (also avoids doubling disk usage), else stream a copy.
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)
    if not dest.exists():  # never hand back a path we cannot actually open
        raise RuntimeError(
            f"SAM3 checkpoint staging produced an unusable path at {dest}."
        )
    return dest
