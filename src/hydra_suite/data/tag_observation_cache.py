"""
Tag observation cache for AprilTag identity signals.

Stores per-frame AprilTag detections as an NPZ sidecar alongside the detection
cache. Each frame records which detection-slot a tag was found in, the tag ID,
the tag center (absolute frame coordinates), and the 4-corner polygon.

Mirrors the API of ``DetectionCache`` so the tracking worker can treat both
caches symmetrically.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public dataclass-style container (plain dict alternative kept intentionally
# lightweight so Numba/serialisation stays trivial).
# ---------------------------------------------------------------------------

# Per-tag observation stored inside a frame:
#   tag_id       : int
#   center_xy    : (float, float)   absolute frame coords
#   corners      : np.ndarray (4,2) absolute frame coords
#   det_index    : int               index into the detection list for that frame
#   hamming      : int               bit-error count from the decoder


class TagObservationCache:
    """NPZ-backed per-frame tag observation cache.

    Write mode (``mode='w'``):
        Call :meth:`add_frame` for every processed frame, then :meth:`save`.

    Read mode (``mode='r'``):
        Instantiate with the path to an existing ``.npz`` file.
        Call :meth:`get_frame` to retrieve observations per frame.
    """

    # Current on-disk format version.  Bump when the key schema changes.
    _FORMAT_VERSION = "1.0"

    def __init__(
        self,
        cache_path: str | Path,
        mode: str = "w",
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ):
        self.cache_path = Path(cache_path)
        self.mode = mode
        self._data: Dict[str, np.ndarray] = {}
        self._loaded_data = None
        self._total_frames = 0
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._cached_frames: Optional[Set[int]] = None
        self._compatible = True
        self._metadata: Dict[str, Any] = {}

        if mode == "r" and self.cache_path.exists():
            logger.info("Loading tag observation cache from %s", self.cache_path)
            self._loaded_data = np.load(str(self.cache_path), allow_pickle=True)
            if "metadata" in self._loaded_data:
                self._metadata = dict(self._loaded_data["metadata"].item())
                cache_version = str(self._metadata.get("version", ""))
                if cache_version != self._FORMAT_VERSION:
                    logger.warning(
                        "Incompatible tag cache version '%s' (expected '%s'). "
                        "Cache will be regenerated.",
                        cache_version,
                        self._FORMAT_VERSION,
                    )
                    self._compatible = False
                    self._loaded_data.close()
                    self._loaded_data = None
                    self._cached_frames = set()
                    return
                self._total_frames = self._metadata.get("total_frames", 0)
                self._start_frame = self._metadata.get("start_frame", 0)
                self._end_frame = self._metadata.get(
                    "end_frame", self._total_frames - 1
                )
            else:
                logger.warning("Tag cache missing metadata – will be regenerated.")
                self._compatible = False
                self._loaded_data.close()
                self._loaded_data = None
                self._cached_frames = set()
                return
            self._cached_frames = self._extract_cached_frames()
            logger.info(
                "Tag cache loaded: %d frames (range %d–%d)",
                self._total_frames,
                self._start_frame,
                self._end_frame,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_cached_frames(self) -> Set[int]:
        if self._loaded_data is None:
            return set()
        cached: Set[int] = set()
        for key in self._loaded_data.files:
            if key.startswith("frame_") and key.endswith("_tag_ids"):
                try:
                    cached.add(int(key.split("_")[1]))
                except (IndexError, ValueError):
                    continue
        return cached

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def add_frame(
        self,
        frame_idx: int,
        tag_ids: Sequence[int],
        centers_xy: Sequence[Tuple[float, float]],
        corners: Sequence[np.ndarray],
        det_indices: Sequence[int],
        hammings: Optional[Sequence[int]] = None,
    ) -> None:
        """Record tag observations for *frame_idx*.

        All sequences must be the same length (one entry per detected tag).
        An empty call (``tag_ids=[]``) is valid and records "no tags in frame".
        """
        if self.mode != "w":
            raise RuntimeError("Cache opened in read mode, cannot write")

        n = len(tag_ids)
        frame_key = f"frame_{frame_idx:06d}"

        if n == 0:
            self._data[f"{frame_key}_tag_ids"] = np.array([], dtype=np.int32)
            self._data[f"{frame_key}_tag_xy"] = np.zeros((0, 2), dtype=np.float32)
            self._data[f"{frame_key}_corners"] = np.zeros((0, 4, 2), dtype=np.float32)
            self._data[f"{frame_key}_det_indices"] = np.array([], dtype=np.int32)
            self._data[f"{frame_key}_hammings"] = np.array([], dtype=np.int32)
        else:
            self._data[f"{frame_key}_tag_ids"] = np.asarray(tag_ids, dtype=np.int32)
            self._data[f"{frame_key}_tag_xy"] = np.asarray(
                centers_xy, dtype=np.float32
            ).reshape(n, 2)
            corners_arr = np.asarray(corners, dtype=np.float32)
            if corners_arr.ndim == 2:
                corners_arr = corners_arr.reshape(n, 4, 2)
            self._data[f"{frame_key}_corners"] = corners_arr
            self._data[f"{frame_key}_det_indices"] = np.asarray(
                det_indices, dtype=np.int32
            )
            _h = hammings if hammings is not None else [0] * n
            self._data[f"{frame_key}_hammings"] = np.asarray(_h, dtype=np.int32)

        self._total_frames = max(self._total_frames, frame_idx + 1)

    def save(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Persist cache to disk as compressed NPZ."""
        if self.mode != "w":
            raise RuntimeError("Cache opened in read mode, cannot save")

        meta: Dict[str, Any] = {
            "total_frames": self._total_frames,
            "start_frame": self._start_frame,
            "end_frame": (
                self._end_frame
                if self._end_frame is not None
                else self._total_frames - 1
            ),
            "version": self._FORMAT_VERSION,
            "format": "tag_observations",
        }
        if metadata:
            meta.update(metadata)
        self._data["metadata"] = np.array(meta)

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(self.cache_path), **self._data)
        file_size_mb = self.cache_path.stat().st_size / (1024 * 1024)
        logger.info("Tag observation cache saved: %.2f MB", file_size_mb)
        self._data.clear()

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_frame(self, frame_idx: int) -> Dict[str, Any]:
        """Return tag observations for *frame_idx*.

        Returns a dict with keys ``tag_ids``, ``centers_xy``, ``corners``,
        ``det_indices``, ``hammings`` — each a numpy array (possibly empty).
        """
        if self.mode != "r":
            raise RuntimeError("Cache opened in write mode, cannot read")
        if self._loaded_data is None:
            raise RuntimeError("No tag cache data loaded")

        frame_key = f"frame_{frame_idx:06d}"
        tag_ids = self._loaded_data.get(
            f"{frame_key}_tag_ids", np.array([], dtype=np.int32)
        )
        centers_xy = self._loaded_data.get(
            f"{frame_key}_tag_xy", np.zeros((0, 2), dtype=np.float32)
        )
        corners = self._loaded_data.get(
            f"{frame_key}_corners", np.zeros((0, 4, 2), dtype=np.float32)
        )
        det_indices = self._loaded_data.get(
            f"{frame_key}_det_indices", np.array([], dtype=np.int32)
        )
        hammings = self._loaded_data.get(
            f"{frame_key}_hammings", np.array([], dtype=np.int32)
        )
        return {
            "tag_ids": tag_ids,
            "centers_xy": centers_xy,
            "corners": corners,
            "det_indices": det_indices,
            "hammings": hammings,
        }

    # ------------------------------------------------------------------
    # Metadata / compatibility queries
    # ------------------------------------------------------------------

    def is_compatible(self) -> bool:
        """Return whether the loaded cache matches the current detector and video settings."""
        return self._compatible

    def get_frame_range(self) -> Tuple[int, int]:
        """Return the (start_frame, end_frame) range stored in the cache."""
        return self._start_frame, self._end_frame if self._end_frame is not None else 0

    def covers_frame_range(self, start_frame: int, end_frame: int) -> bool:
        """Check whether every frame in [start_frame, end_frame] has cached observations."""
        if self._loaded_data is None or self._cached_frames is None:
            return False
        if self._start_frame > start_frame or (
            self._end_frame is not None and self._end_frame < end_frame
        ):
            return False
        return all(f in self._cached_frames for f in range(start_frame, end_frame + 1))

    def get_total_frames(self) -> int:
        """Return the number of frames with cached tag observations."""
        return self._total_frames

    def get_cached_frames(self) -> list[int]:
        """Return the sorted frame indices that have cached tag observations."""
        if self.mode == "r":
            return sorted(self._cached_frames or set())
        cached: Set[int] = set()
        for key in self._data:
            if key.startswith("frame_") and key.endswith("_tag_ids"):
                try:
                    cached.add(int(key.split("_")[1]))
                except (IndexError, ValueError):
                    continue
        return sorted(cached)

    def get_metadata(self) -> Dict[str, Any]:
        """Return a copy of the cache metadata dict (detector settings, version, etc.)."""
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying .npz memory-mapped file handle."""
        if self._loaded_data is not None:
            self._loaded_data.close()
            self._loaded_data = None
        logger.info("Tag observation cache closed: %s", self.cache_path)

    def __enter__(self):
        return self

    def __exit__(self, _et, _ev, _tb):
        self.close()


# ---------------------------------------------------------------------------
# Utility: compute a short hash of a detection cache file for staleness check
# ---------------------------------------------------------------------------


def detection_cache_hash(
    detection_cache_path: str | Path, chunk_size: int = 65536
) -> str:
    """Return a hex digest (SHA-256, first 16 chars) of a detection cache file."""
    h = hashlib.sha256()
    with open(detection_cache_path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]
