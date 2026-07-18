from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    repo_id: str
    filename: str
    sha256: str
    variant: str
    num_keypoints: int
    description: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_pinned(repo_id: str, filename: str, sha256: str, dest: Path) -> Path:
    """Download via HF hub (mirrors tools/vitpose/fetch_assets.py) and verify the
    pinned SHA256. Atomic: a hash mismatch leaves no file at `dest`."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and _sha256(dest) == sha256:
        return dest
    src = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(src.read_bytes())
    got = _sha256(tmp)
    if got != sha256:
        os.remove(tmp)
        raise ValueError(
            f"SHA256 mismatch for {repo_id}/{filename}\n  expected {sha256}\n  got      {got}"
        )
    os.replace(tmp, dest)
    return dest


# COCO pins are the real values from tools/vitpose/fetch_assets.py (already
# Spec-1-validated). Add L/H and the AP-10K/APT-36K animal entries the same way:
# a real (repo_id, filename, sha256) plus a backbone-strict load test (below)
# BEFORE the entry ships. If an animal asset cannot be pinned, omit it (COCO-only
# catalog + Browse is the spec's accepted fallback).
CATALOG: dict[str, CatalogEntry] = {
    "vitpose-b-coco": CatalogEntry(
        name="ViTPose-B (COCO)",
        repo_id="nielsr/vitpose-original-checkpoints",
        filename="vitpose-b.pth",
        sha256="2e849e1f1dbb5b87191eda7171f1b16468d5d082a7380e93e94b7ce76a061679",
        variant="B",
        num_keypoints=17,
        description="ViTPose-B, human COCO-17. General-purpose start.",
    ),
    # "vitpose-l-coco", "vitpose-h-coco", "vitpose-b-ap10k" (animal), ... added likewise.
}


def resolve_checkpoint(name_or_path: str, cache_dir: Path) -> Path:
    if name_or_path in CATALOG:
        e = CATALOG[name_or_path]
        return fetch_pinned(
            e.repo_id, e.filename, e.sha256, Path(cache_dir) / f"{name_or_path}.pth"
        )
    p = Path(name_or_path)
    if p.exists():
        return p
    raise ValueError(f"not a catalog name or existing file: {name_or_path!r}")
