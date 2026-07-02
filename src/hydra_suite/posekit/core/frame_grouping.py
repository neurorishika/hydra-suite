"""Group PoseKit image indices into per-source frames.

Reuses the MAT/FilterKit identity-crop filename convention
(``did<detection_id>.<ext>``, where ``frame_idx = detection_id // 10000``)
so images produced by a FilterKit export can be regrouped by their
original source frame.
"""

from __future__ import annotations

from typing import Any, Sequence

from hydra_suite.core.identity.dataset.naming import parse_identity_image_filename


def group_indices_by_frame(
    filenames: Sequence[str], source_ids: Sequence[Any]
) -> dict[tuple[Any, int], list[int]]:
    """Group image indices by ``(source_id, frame_idx)``.

    ``filenames[i]``/``source_ids[i]`` describe the image at global index
    ``i``. Filenames that don't match the identity-crop convention each
    get a unique singleton key ``(source_id, -(i + 1))`` — real
    ``frame_idx`` values are always >= 0, so singleton keys never collide
    with a genuine frame.
    """
    groups: dict[tuple[Any, int], list[int]] = {}
    for idx, (filename, source_id) in enumerate(zip(filenames, source_ids)):
        parsed = parse_identity_image_filename(filename)
        frame_component = parsed["frame_idx"] if parsed is not None else -(idx + 1)
        key = (source_id, frame_component)
        groups.setdefault(key, []).append(idx)
    return groups
