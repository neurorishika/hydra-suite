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


def test_resume_restores_lr_scheduler_and_metrics_header(tmp_path):
    # Verifies the review-round-1 fix: last.pt now carries the scheduler state so
    # a resumed run continues the cosine curve instead of restarting it, and a
    # resume into a fresh output_dir still writes a metrics.csv header.
    ds = tmp_path / "ds"
    _tiny_dataset(ds)
    from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose

    pre = tmp_path / "pre.pth"
    torch.save({"state_dict": build_vitpose("S", "classic", 3).state_dict()}, pre)

    out = tmp_path / "run"
    cfg = RunConfig(
        init_checkpoint=str(pre),
        variant="S",
        num_keypoints=3,
        dataset_dir=str(ds),
        output_dir=str(out),
        device="cpu",
        epochs=4,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        drop_path=0.0,
        seed=0,
    )
    train(cfg)

    last = out / "last.pt"
    assert last.exists()
    ckpt = torch.load(last, map_location="cpu", weights_only=True)
    assert "sched_state" in ckpt

    # The saved scheduler state reflects the completed 4-epoch run (construction
    # steps once, then epochs 0..3 each call sched.step() once more), i.e.
    # last_epoch == 4 -- not a scheduler restarted from last_epoch == -1.
    probe_model = build_vitpose("S", "classic", 3)
    probe_opt = torch.optim.AdamW(probe_model.parameters(), lr=1e-3)
    probe_sched = torch.optim.lr_scheduler.CosineAnnealingLR(probe_opt, T_max=4)
    probe_sched.load_state_dict(ckpt["sched_state"])
    assert probe_sched.last_epoch == 4

    # Resume into a brand-new output_dir (no pre-existing metrics.csv) for one more
    # epoch; the header guard must still fire even though start_epoch > 0.
    resume_out = tmp_path / "resume_run"
    resume_cfg = RunConfig(
        init_checkpoint=str(pre),
        variant="S",
        num_keypoints=3,
        dataset_dir=str(ds),
        output_dir=str(resume_out),
        device="cpu",
        epochs=5,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        drop_path=0.0,
        seed=0,
        resume_from=str(last),
    )
    train(resume_cfg)

    resumed_metrics = resume_out / "metrics.csv"
    assert resumed_metrics.exists()
    lines = resumed_metrics.read_text().strip().splitlines()
    assert lines[0] == "epoch,train_loss,val_loss,pck@0.05,pck@0.1"
    # Only epoch 4 ran (resumed at start_epoch=4, epochs=5).
    assert len(lines) == 2
    assert lines[1].split(",")[0] == "4"


def test_fresh_rerun_overwrites_stale_metrics_csv(tmp_path):
    # Regression guard: a non-resume run (start_epoch == 0) into an output_dir
    # that already has a metrics.csv from a prior run must truncate + rewrite
    # the header, not append onto the stale file (which would produce a
    # headerless file with duplicate epoch numbers).
    ds = tmp_path / "ds"
    _tiny_dataset(ds)
    from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose

    pre = tmp_path / "pre.pth"
    torch.save({"state_dict": build_vitpose("S", "classic", 3).state_dict()}, pre)

    out = tmp_path / "run"
    cfg = RunConfig(
        init_checkpoint=str(pre),
        variant="S",
        num_keypoints=3,
        dataset_dir=str(ds),
        output_dir=str(out),
        device="cpu",
        epochs=2,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        drop_path=0.0,
        seed=0,
    )
    train(cfg)

    metrics_path = out / "metrics.csv"
    assert metrics_path.exists()
    first_run_lines = metrics_path.read_text().strip().splitlines()
    assert len(first_run_lines) == 3  # header + 2 epoch rows

    # Rerun fresh (no resume_from) into the SAME output_dir.
    train(cfg)

    lines = metrics_path.read_text().strip().splitlines()
    assert lines[0] == "epoch,train_loss,val_loss,pck@0.05,pck@0.1"
    assert lines.count(lines[0]) == 1  # exactly one header row
    assert len(lines) == 3  # header + 2 epoch rows, no stale duplicates
    assert lines[1].split(",")[0] == "0"
    assert lines[2].split(",")[0] == "1"
