"""ViTPose training recipe: augmentation, split, EMA, schedule, validation, UI."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.core.individual.pose.vitpose.training import augment as A  # noqa: E402
from hydra_suite.core.individual.pose.vitpose.training.config import (  # noqa: E402
    validate_run_config,
)
from hydra_suite.core.individual.pose.vitpose.training.split import (  # noqa: E402
    group_key,
    split_ids,
)

BASE = dict(
    init_checkpoint="x.pth",
    variant="B",
    num_keypoints=4,
    dataset_dir="d",
    output_dir="o",
)


# ── augmentation ─────────────────────────────────────────────────────────


def _img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)


def test_pixel_augmentation_is_identity_when_off():
    params = A.AugmentParams(brightness=0, contrast=0, rotation=0)
    img = _img()
    out = A.augment_pixels(img, params, np.random.default_rng(1))
    np.testing.assert_array_equal(out, img)
    assert A.sample_rotation(params, np.random.default_rng(1)) == 0.0


def test_each_pixel_augmentation_changes_the_image():
    img = _img()
    for kw in (
        dict(saturation=0.5),
        dict(hue=0.2),
        dict(blur_prob=1.0),
        dict(noise_prob=1.0),
        dict(occlusion_prob=1.0),
    ):
        params = A.AugmentParams(brightness=0, contrast=0, **kw)
        out = A.augment_pixels(img, params, np.random.default_rng(3))
        assert out.shape == img.shape and out.dtype == np.uint8
        assert not np.array_equal(out, img), kw


def test_occlusion_is_one_rectangle_of_bounded_size():
    img = np.zeros((100, 100, 3), np.uint8)
    params = A.AugmentParams(brightness=0, contrast=0, occlusion_prob=1.0)
    for seed in range(20):
        out = A.occlude(img, params, np.random.default_rng(seed))
        ys, xs = np.nonzero(out.any(axis=2))
        if ys.size == 0:  # a black patch on black: nothing visible
            continue
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        assert 10 <= h <= 40 and 10 <= w <= 40
        assert (out.any(axis=2)).sum() == h * w  # a filled rectangle


def test_rotation_within_range():
    params = A.AugmentParams(rotation=30, rotation_prob=1.0)
    rng = np.random.default_rng(0)
    rots = [A.sample_rotation(params, rng) for _ in range(500)]
    assert max(abs(r) for r in rots) <= 30 and np.std(rots) > 5


# ── split ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name, frame, video",
    [
        ("images/vid-a__f000644__018.png", "vid-a__f000644", "vid-a"),
        ("Mix26__frame_07790__019.png", "Mix26__frame_07790", "Mix26"),
        ("20260506_181426-cf__f000001__000.png", "20260506_181426-cf__f000001", None),
        ("plain.png", "plain", "plain"),
    ],
)
def test_group_keys(name, frame, video):
    assert group_key(name, "frame") == frame
    if video:
        assert group_key(name, "video") == video
    assert group_key(name, "image") == Path(name).stem


def test_split_keeps_frames_together_and_is_deterministic():
    ids = list(range(40))
    names = {i: f"v{i % 4}__f{i // 4}__{i:03d}.png" for i in ids}
    train, val = split_ids(ids, names, 0.25, 7, "frame")
    assert set(train) | set(val) == set(ids) and not set(train) & set(val)
    frames_val = {group_key(names[i], "frame") for i in val}
    frames_train = {group_key(names[i], "frame") for i in train}
    assert not frames_val & frames_train
    assert split_ids(ids, names, 0.25, 7, "frame") == (train, val)
    # one big group never leaves training empty
    one = {i: f"v__f1__{i:03d}.png" for i in range(5)}
    one[5] = "v__f2__000.png"
    tr, va = split_ids(list(range(6)), one, 0.9, 0, "frame")
    assert tr and va


# ── config ───────────────────────────────────────────────────────────────


def test_old_run_json_keeps_legacy_recipe():
    cfg = validate_run_config(dict(BASE))
    assert (cfg.rotation, cfg.rotation_prob, cfg.brightness, cfg.contrast) == (
        40.0,
        0.6,
        0.3,
        0.3,
    )
    assert not cfg.ema and not cfg.amp and cfg.split_by == "image"
    assert cfg.blur_prob == cfg.noise_prob == cfg.occlusion_prob == 0.0


@pytest.mark.parametrize(
    "bad",
    [
        {"occlusion_prob": 1.5},
        {"hue": "x"},
        {"rotation": 200},
        {"warmup_epochs": -1},
        {"split_by": "random"},
        {"layer_decay": 2.0},
        {"amp": True, "ema_decay": 1.5},
    ],
)
def test_run_config_rejects_bad_recipe(bad):
    with pytest.raises(ValueError):
        validate_run_config(dict(BASE, **bad))


def test_warmup_cosine_schedule():
    from hydra_suite.core.individual.pose.vitpose.training.train import (
        _warmup_cosine,
    )

    f = _warmup_cosine(100, 10)
    assert f(0) == pytest.approx(0.001)
    assert f(5) < f(9) < f(10) == pytest.approx(1.0)
    assert f(55) == pytest.approx(0.5, abs=0.02)
    assert f(100) == pytest.approx(0.0, abs=1e-6)
    assert _warmup_cosine(10, 0)(0) == pytest.approx(1.0)


# ── EMA ──────────────────────────────────────────────────────────────────


def test_ema_tracks_the_model():
    from hydra_suite.core.individual.pose.vitpose.training.ema import ModelEMA

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():
        model[0].weight.fill_(1.0)
    for _ in range(20000):
        ema.update(model)
    assert torch.allclose(ema.module[0].weight, torch.ones(2, 2), atol=1e-3)
    assert not any(p.requires_grad for p in ema.module.parameters())
    clone = ModelEMA(model)
    clone.load_state_dict(ema.state_dict())
    assert clone.updates == 20000


# ── dataset + validation ─────────────────────────────────────────────────


def _coco(tmp_path: Path, n=6):
    import cv2

    ds = tmp_path / "coco"
    (ds / "images").mkdir(parents=True)
    images, anns = [], []
    for i in range(n):
        img = np.full((48, 64, 3), 40, np.uint8)
        cv2.circle(img, (20, 24), 4, (255, 255, 255), -1)
        cv2.imwrite(str(ds / "images" / f"v__f{i}__000.png"), img)
        images.append(
            {
                "id": i + 1,
                "file_name": f"images/v__f{i}__000.png",
                "width": 64,
                "height": 48,
            }
        )
        anns.append(
            {
                "id": i + 1,
                "image_id": i + 1,
                "category_id": 1,
                "keypoints": [20.0, 24.0, 2, 44.0, 24.0, 1],
                "num_keypoints": 2,
                "bbox": [10, 10, 40, 30],
            }
        )
    coco = {"images": images, "annotations": anns, "categories": [{"id": 1}]}
    (ds / "annotations.json").write_text(json.dumps(coco))
    return ds


def test_dataset_ignore_occluded_zeroes_weight(tmp_path):
    from hydra_suite.core.individual.pose.vitpose.training.dataset import (
        CocoKeypointsDataset,
    )

    ds_dir = _coco(tmp_path)
    plain = CocoKeypointsDataset(ds_dir, [1], 2.0, augment=False)
    ignored = CocoKeypointsDataset(ds_dir, [1], 2.0, augment=True, ignore_occluded=True)
    assert plain[0]["target_weight"][1].item() > 0  # v=1 counts by default
    assert ignored[0]["target_weight"][1].item() == 0
    assert ignored[0]["target_weight"][0].item() > 0


def test_dataset_reseed_gives_independent_streams(tmp_path):
    from hydra_suite.core.individual.pose.vitpose.training.dataset import (
        CocoKeypointsDataset,
    )

    ds_dir = _coco(tmp_path)
    aug = A.AugmentParams(noise_prob=1.0, brightness=0.3)
    a = CocoKeypointsDataset(ds_dir, [1], 2.0, augment=True, aug=aug, seed=1)
    b = CocoKeypointsDataset(ds_dir, [1], 2.0, augment=True, aug=aug, seed=1)
    assert torch.equal(a[0]["image"], b[0]["image"])
    b.reseed(2)
    assert not torch.equal(a[0]["image"], b[0]["image"])


def test_flip_test_validation_runs_and_reports_nme(tmp_path):
    from torch.utils.data import DataLoader

    from hydra_suite.core.individual.pose.vitpose.training.dataset import (
        CocoKeypointsDataset,
    )
    from hydra_suite.core.individual.pose.vitpose.training.validate import (
        run_validation,
    )
    from hydra_suite.core.individual.pose.vitpose.vitpose import build_vitpose

    ds = CocoKeypointsDataset(_coco(tmp_path), [1, 2], 2.0, augment=False)
    model = build_vitpose("S", "classic", 2)
    loader = DataLoader(ds, batch_size=2)
    plain = run_validation(model, loader, torch.device("cpu"))
    flipped = run_validation(model, loader, torch.device("cpu"), flip_pairs=[(0, 1)])
    assert np.isfinite(plain["nme"]) and np.isfinite(flipped["nme"])
    assert plain["val_loss"] == pytest.approx(flipped["val_loss"])


# ── UI: recipe panel + worker hand-off ───────────────────────────────────


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_recipe_panel_params_validate_and_round_trip(qapp):
    from hydra_suite.posekit.gui.panels.vitpose_recipe_panel import (
        RECOMMENDED,
        ViTPoseRecipePanel,
    )

    panel = ViTPoseRecipePanel()
    params = panel.params()
    for key, value in RECOMMENDED.items():
        assert (
            params[key] == pytest.approx(value)
            if isinstance(value, float)
            else (params[key] == value)
        ), key
    cfg = validate_run_config(dict(BASE, **params))  # every key is a RunConfig key
    assert cfg.split_by == "frame" and cfg.ema and cfg.amp
    assert cfg.layer_decay is None and cfg.drop_path is None  # "Auto"

    panel.apply_settings(
        {"occlusion_prob": 0.7, "split_by": "video", "layer_decay": 0.6, "ema": False}
    )
    again = ViTPoseRecipePanel()
    again.apply_settings(panel.settings())
    p = again.params()
    assert p["occlusion_prob"] == pytest.approx(0.7) and p["split_by"] == "video"
    assert p["layer_decay"] == pytest.approx(0.6) and p["ema"] is False
    # malformed values keep the current ones
    again.apply_settings({"lr": "abc", "ema": "yes", "split_by": "nope"})
    assert again.params()["lr"] == pytest.approx(p["lr"])
    again.reset_recommended()
    assert again.params()["occlusion_prob"] == RECOMMENDED["occlusion_prob"]


def test_worker_passes_recipe_and_aux(monkeypatch, tmp_path):
    import hydra_suite.posekit.gui.dialogs.training as T

    seen = {}
    monkeypatch.setattr(
        T,
        "build_coco_keypoints_dataset",
        lambda **kw: seen.update(build=kw) or {"dataset_dir": tmp_path / "ds"},
    )
    monkeypatch.setattr(
        T,
        "prepare_run",
        lambda params, run_dir, cache_dir: seen.update(params=params)
        or Path(run_dir) / "run.json",
    )
    monkeypatch.setattr(T, "build_training_command", lambda rj: ["true"])

    class _Proc:
        stdout = iter(())

        def wait(self):
            return 0

    monkeypatch.setattr(T.subprocess, "Popen", lambda *a, **k: _Proc())
    aux = [([tmp_path / "a.png"], tmp_path / "labels")]
    w = T.ViTPoseTrainingWorker(
        image_paths=[],
        labels_dir=tmp_path,
        run_dir=tmp_path / "run",
        cache_dir=tmp_path / "c",
        class_names=["a"],
        keypoint_names=["k"],
        skeleton_edges=[],
        variant="B",
        init_checkpoint="vitpose-b-ap10k",
        num_keypoints=1,
        epochs=1,
        batch=1,
        device="cpu",
        recipe={"occlusion_prob": 0.4, "split_by": "frame", "ema": True},
        extra_datasets=aux,
    )
    w.run()
    assert seen["params"]["occlusion_prob"] == 0.4
    assert seen["params"]["split_by"] == "frame" and seen["params"]["ema"]
    assert seen["build"]["extra_datasets"] == aux


def test_catalog_checkpoint_fixes_the_variant(qapp, tmp_path, monkeypatch):
    import hydra_suite.posekit.gui.utils as gu
    from hydra_suite.posekit.gui.dialogs.training import TrainingRunnerDialog
    from hydra_suite.posekit.gui.models import Project

    monkeypatch.setattr(gu, "get_ui_settings_path", lambda: tmp_path / "ui.json")
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    proj = Project(
        images_dir=tmp_path / "images",
        out_root=tmp_path,
        labels_dir=tmp_path / "labels",
        project_path=tmp_path / "pose_project.json",
        class_names=["a"],
        keypoint_names=["head", "tail"],
        skeleton_edges=[(0, 1)],
    )
    dlg = TrainingRunnerDialog(None, proj, [])
    assert dlg.vitpose_recipe.isVisibleTo(dlg)  # ViTPose is the default backend
    dlg.vitpose_checkpoint_combo.setCurrentText("vitpose-l-ap10k")
    assert dlg.vitpose_variant_combo.currentText() == "L"
    assert not dlg.vitpose_variant_combo.isEnabled()
    dlg.vitpose_checkpoint_combo.setCurrentText("/my/own.pth")
    assert dlg.vitpose_variant_combo.isEnabled()
    dlg._loss_timer.stop()


def test_split_single_group_falls_back_to_images():
    names = {i: f"onlyvideo__f{i}__000.png" for i in range(50)}
    train, val = split_ids(list(range(50)), names, 0.2, 0, "video")
    assert len(train) == 40 and len(val) == 10


def test_split_does_not_overshoot_with_a_huge_group():
    names = {
        i: (f"A__f{i}__000.png" if i < 900 else f"B__f{i}__000.png")
        for i in range(1000)
    }
    for seed in range(5):
        train, val = split_ids(list(range(1000)), names, 0.2, seed, "video")
        assert len(val) == 100 and len(train) == 900


def test_group_key_ignores_collision_suffix():
    assert group_key("v__f1__000_ab12cd34.png", "frame") == "v__f1"


@pytest.mark.parametrize(
    "bad",
    [
        {"amp": "false"},
        {"ema": 1},
        {"num_workers": 2.5},
        {"patience": True},
        {"ema_decay": 1.0},
    ],
)
def test_run_config_rejects_wrong_types(bad):
    with pytest.raises(ValueError):
        validate_run_config(dict(BASE, **bad))


def test_single_video_split_keeps_frames_together():
    names = {i: f"vid__f{i // 5}__{i % 5:03d}.png" for i in range(50)}  # 10 frames
    train, val = split_ids(list(range(50)), names, 0.2, 0, "video")
    assert train and val
    frames = lambda ids: {group_key(names[i], "frame") for i in ids}  # noqa: E731
    assert not frames(train) & frames(val)


def test_decode_handles_a_single_keypoint():
    from hydra_suite.core.individual.pose.vitpose.decode import decode_udp_cv2

    rng = np.random.default_rng(0)
    coords, maxvals = decode_udp_cv2(rng.random((3, 1, 16, 12)), kernel=3)
    assert coords.shape == (3, 1, 2) and maxvals.shape == (3, 1, 1)


def test_ema_decay_null_is_a_value_error():
    with pytest.raises(ValueError):
        validate_run_config(dict(BASE, ema_decay=None))
