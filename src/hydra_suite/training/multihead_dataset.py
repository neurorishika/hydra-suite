"""Composite-folder dataset for shared-trunk multi-head training.

Folder layout:
    <root>/<f0_label><DELIM><f1_label>[<DELIM><f2_label>...]/<file>.<ext>

Each composite folder name decomposes into ``K`` factor labels using the
configured delimiter. Files inside a composite folder all share the same
multi-factor label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class MultiFactorImageFolder(Dataset):
    """Yield ``(image_tensor, LongTensor[K])`` for K-factor classification."""

    def __init__(
        self,
        root: str,
        *,
        class_names_per_factor: list[list[str]],
        delimiter: str = "__",
        transform: Callable | None = None,
    ) -> None:
        if not class_names_per_factor:
            raise ValueError("class_names_per_factor must be non-empty")
        if not delimiter:
            raise ValueError("delimiter must be a non-empty string")
        root_path = Path(root)
        if not root_path.is_dir():
            raise ValueError(f"{root!r}: not a directory")

        self._transform = transform
        self._delimiter = delimiter
        self._class_names_per_factor = [list(c) for c in class_names_per_factor]
        n_factors = len(self._class_names_per_factor)
        per_factor_index = [
            {name: idx for idx, name in enumerate(inner)}
            for inner in self._class_names_per_factor
        ]

        self._samples: list[tuple[str, list[int]]] = []
        for entry in sorted(root_path.iterdir()):
            if not entry.is_dir():
                continue
            parts = entry.name.split(delimiter)
            if len(parts) != n_factors:
                raise ValueError(
                    f"{entry.name!r}: factor count mismatch "
                    f"(expected {n_factors} parts split by {delimiter!r}, got {len(parts)})"
                )
            label_tuple: list[int] = []
            for k, part in enumerate(parts):
                if part not in per_factor_index[k]:
                    raise ValueError(
                        f"{entry.name!r}: unknown label {part!r} for factor {k} "
                        f"(allowed: {self._class_names_per_factor[k]!r})"
                    )
                label_tuple.append(per_factor_index[k][part])
            for fp in sorted(entry.iterdir()):
                if fp.is_file() and fp.suffix.lower() in _IMG_EXTENSIONS:
                    self._samples.append((str(fp), label_tuple))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        path, label_tuple = self._samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self._transform is not None:
            tensor = self._transform(image)
        else:
            import numpy as np

            tensor = torch.from_numpy(
                np.asarray(image, dtype="float32") / 255.0
            ).permute(2, 0, 1)
        return tensor, torch.LongTensor(label_tuple)

    @property
    def class_names_per_factor(self) -> list[list[str]]:
        return [list(c) for c in self._class_names_per_factor]
