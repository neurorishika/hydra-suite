"""Fetch and integrity-check ViTPose assets.

Checkpoints come from a third-party re-host (nielsr/vitpose-original-checkpoints)
because upstream publishes OneDrive links only, which 403 to non-browser clients.
Every asset is SHA256-pinned: for weights because we do not control the host, and
for the COCO detections because a plausible-looking dummy is in circulation.
"""

from __future__ import annotations

import hashlib
from collections import namedtuple
from pathlib import Path

Asset = namedtuple("Asset", "kind repo_or_id filename sha256 size")


class AssetIntegrityError(RuntimeError):
    """Raised when a downloaded asset does not match its pinned digest."""


ASSETS: dict[str, Asset] = {
    "vitpose-b": Asset(
        kind="hf",
        repo_or_id="nielsr/vitpose-original-checkpoints",
        filename="vitpose-b.pth",
        sha256="2e849e1f1dbb5b87191eda7171f1b16468d5d082a7380e93e94b7ce76a061679",
        size=360_038_314,
    ),
    "vitpose-b-simple": Asset(
        kind="hf",
        repo_or_id="nielsr/vitpose-original-checkpoints",
        filename="vitpose-b-simple.pth",
        sha256="1d60b1e86af84d36cb57d500c5d517f66beb211cfcb36d5e60c741f2164533da",
        size=343_701_438,
    ),
    "vitpose-plus-base": Asset(
        kind="hf",
        repo_or_id="nielsr/vitpose-original-checkpoints",
        filename="vitpose+_base.pth",
        sha256="de043d2ea25a8bcffeb8ea278a68f52e487b918866b629efb2f894b15fdb3545",
        size=585_832_783,
    ),
    "coco_val2017_person_detections": Asset(
        kind="gdrive",
        repo_or_id="1ygw57X-mh0QBfENB-U5DsuSauGIu-8RB",
        filename="COCO_val2017_detections_AP_H_56_person.json",
        sha256="53ba0ad8d0fd461c5a000cd90797fa8c39cd8c38cd125125c0412626ff592d59",
        size=16_383_781,
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, expected_sha256: str) -> None:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise AssetIntegrityError(
            f"sha256 mismatch for {path.name}: "
            f"expected {expected_sha256}, got {actual}"
        )


def fetch(name: str, dest_dir: Path, allow_unpinned: bool = False) -> Path:
    """Download (if absent) and verify an asset.

    An unpinned asset (sha256="") raises unless allow_unpinned=True. Silently
    skipping verification for unpinned entries would defeat the point of the
    module: the bootstrap that DISCOVERS a digest must say so explicitly.
    Only the Step 6 bootstrap passes allow_unpinned=True.
    """
    asset = ASSETS[name]
    if not asset.sha256 and not allow_unpinned:
        raise AssetIntegrityError(
            f"{name} has no pinned sha256; run the Step 6 bootstrap to record "
            f"one, or pass allow_unpinned=True if you are that bootstrap"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / asset.filename
    if not out.exists():
        if asset.kind == "hf":
            from huggingface_hub import hf_hub_download

            src = hf_hub_download(repo_id=asset.repo_or_id, filename=asset.filename)
            out.write_bytes(Path(src).read_bytes())
        elif asset.kind == "gdrive":
            import gdown

            gdown.download(id=asset.repo_or_id, output=str(out), quiet=False)
        else:
            raise ValueError(f"unknown asset kind: {asset.kind}")
    if asset.sha256:
        verify(out, asset.sha256)
    if asset.size and out.stat().st_size != asset.size:
        raise AssetIntegrityError(
            f"size mismatch for {asset.filename}: "
            f"expected {asset.size}, got {out.stat().st_size}"
        )
    return out
