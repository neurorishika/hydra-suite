import json
from pathlib import Path

import cv2
import numpy as np
import torch

from hydra_suite.core.identity.pose.vitpose.training.config import RunConfig
from hydra_suite.core.identity.pose.vitpose.training.train import train


def _tiny_dataset(root: Path, n=8, k=3):
    (root / "images").mkdir(parents=True)
    images, anns = [], []
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, (100, 80, 3), dtype=np.uint8)
        cv2.imwrite(str(root / "images" / f"f{i}.png"), img)
        kp = []
        for j in range(k):
            kp += [25 + 4 * j, 30 + 4 * j, 2]
        images.append(
            {"id": i + 1, "file_name": f"f{i}.png", "width": 80, "height": 100}
        )
        anns.append(
            {
                "id": i + 1,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 50.0, 70.0],
                "area": 3500.0,
                "iscrowd": 0,
                "num_keypoints": k,
                "keypoints": kp,
            }
        )
    coco = {
        "images": images,
        "annotations": anns,
        "categories": [
            {
                "id": 1,
                "name": "a",
                "keypoints": [f"k{j}" for j in range(k)],
                "skeleton": [],
            }
        ],
    }
    (root / "annotations.json").write_text(json.dumps(coco))


def test_tiny_overfit_drives_metrics(tmp_path):
    # Variant "S" (not "B") keeps this CPU gate fast; the loop/targets/loss are
    # architecture-agnostic, so S exercises exactly the same code paths.
    ds = tmp_path / "ds"
    _tiny_dataset(ds)
    # a random-init "pretrained" checkpoint so the loader path is exercised
    from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose

    pre = tmp_path / "pre.pth"
    torch.save({"state_dict": build_vitpose("S", "classic", 17).state_dict()}, pre)

    out = tmp_path / "run"
    cfg = RunConfig(
        init_checkpoint=str(pre),
        variant="S",
        num_keypoints=3,
        dataset_dir=str(ds),
        output_dir=str(out),
        device="cpu",
        epochs=3,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        drop_path=0.0,
        seed=0,
    )
    train(cfg)
    assert (out / "best.pt").exists()
    assert (out / "metrics.csv").exists()
    # loss must have decreased across the 3 epochs
    rows = (out / "metrics.csv").read_text().strip().splitlines()[1:]
    losses = [float(r.split(",")[1]) for r in rows]
    assert losses[-1] < losses[0]
    # best.pt loads back into a K=3 classic model
    blob = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert blob["num_keypoints"] == 3
    build_vitpose("S", "classic", 3).load_state_dict(blob["model_state"])
