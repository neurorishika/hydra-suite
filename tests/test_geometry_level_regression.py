from pathlib import Path

import cv2
import numpy as np

from hydra_suite.training.contracts import SourceDataset, SplitConfig
from hydra_suite.training.dataset_builders import (
    derive_detect_dataset_from_obb,
    merge_obb_sources,
)


def _obb_source(root: Path):
    for split in ("all",):
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(4):
        # Distinct pixel content per image so the merge step's content-hash
        # dedup (default dedup=True) does not collapse identical frames.
        img = np.full((32, 32, 3), fill_value=i * 40, dtype=np.uint8)
        cv2.imwrite(str(root / "images" / f"f{i}.jpg"), img)
        (root / "labels" / f"f{i}.txt").write_text(
            "0 0.10 0.12 0.51 0.13 0.49 0.55 0.11 0.52\n", encoding="utf-8"
        )
    (root / "classes.txt").write_text("object\n", encoding="utf-8")


def test_obb_only_detect_derivation_is_stable(tmp_path):
    src = tmp_path / "src"
    _obb_source(src)
    merged = merge_obb_sources(
        [SourceDataset(path=str(src), name="s")],
        tmp_path / "merged",
        SplitConfig(0.75, 0.25, 0.0),
        class_names=["object"],
        seed=7,
    )
    detect = derive_detect_dataset_from_obb(
        merged.dataset_dir, tmp_path / "det", class_names=["object"]
    )
    # Golden AABB for the OBB above: x in [0.10,0.51], y in [0.12,0.55].
    for lbl in (Path(detect.dataset_dir) / "labels").rglob("*.txt"):
        cx, cy, bw, bh = (float(v) for v in lbl.read_text().split()[1:])
        assert abs(bw - 0.41) < 1e-4 and abs(bh - 0.43) < 1e-4
        assert abs(cx - 0.305) < 1e-4 and abs(cy - 0.335) < 1e-4
