"""Image scanning, hashing, and database ingestion utilities."""

import hashlib
from pathlib import Path
from typing import Any, Dict, Generator, List, Union


def compute_image_hash(path: Path) -> str:
    """Compute MD5 hash of an image file."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def scan_images(
    root: Union[str, Path],
    extensions: List[str] = None,
    images_subdir: str = "images",
) -> Generator[Path, None, None]:
    """Yield image paths from *root* or its ``images/`` subdirectory.

    Preference order (matching PoseKit convention):
    1. If ``root/images/`` exists and contains images, scan **only** that
       directory (non-recursively).
    2. Otherwise scan ``root`` itself (non-recursively).

    This intentionally avoids deep recursive scans so that project metadata
    folders, label outputs, and preview caches are never accidentally ingested.
    """
    root = Path(root)
    if not root.exists():
        return

    if extensions is None:
        extensions = [".jpg", ".png", ".jpeg"]

    valid_exts = {e.lower() for e in extensions}

    # Prefer the images/ subdirectory when it contains images.
    images_dir = root / images_subdir
    if images_dir.is_dir():
        candidates = [
            p.resolve()
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ]
        if candidates:
            yield from iter(candidates)
            return

    # Fall back to the root folder itself (non-recursive).
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in valid_exts:
            yield path.resolve()


class IngestWorker:
    """Worker for scanning and ingesting images into database."""

    def __init__(self, db):
        """
        Args:
            db: ClassKitDB instance
        """
        self.db = db

    def ingest(
        self,
        image_paths: List[Path],
        compute_hashes: bool = True,
        metadata_by_path: Dict[str, Dict[str, Any]] | None = None,
    ):
        """Ingest images into database.

        Args:
            image_paths: List of image paths
            compute_hashes: Whether to compute MD5 hashes
        """
        if compute_hashes:
            hashes = [compute_image_hash(p) for p in image_paths]
        else:
            hashes = None

        self.db.add_images(image_paths, hashes, metadata_by_path=metadata_by_path)
