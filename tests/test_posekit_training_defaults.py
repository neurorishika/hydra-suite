"""PoseKit training: flip pairs, mosaic, crop-size auto-sizing, ViTPose default."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ANT = [
    "left_antenna_tip",
    "right_antenna_tip",
    "left_antenna_elbow",
    "right_antenna_elbow",
    "clypeus",
    "neck",
]


# ── flip plan ────────────────────────────────────────────────────────────


def test_plan_flips_cases():
    from hydra_suite.posekit.core.training_options import plan_flips

    off = plan_flips(ANT, [(0, 1)], 0.0, 0.0)
    assert not off.enabled and off.flip_idx is None

    paired = plan_flips(ANT, [(0, 1), (2, 3)], 0.5, 0.25)
    assert (paired.fliplr, paired.flipud) == (0.5, 0.25)
    assert paired.flip_idx == [1, 0, 3, 2, 4, 5]

    # left/right names but no pairs: flipping would swap sides -> off
    blocked = plan_flips(ANT, [], 0.5, 0.5)
    assert blocked.blocked and not blocked.enabled and blocked.flip_idx is None

    midline = plan_flips(["head", "thorax", "tail"], [], 0.5, 0.0)
    assert midline.enabled and midline.flip_idx == [0, 1, 2]


# ── auto sizing from crops ───────────────────────────────────────────────


def _write_images(folder: Path, sizes):
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (w, h) in enumerate(sizes):
        p = folder / f"img_{i:03d}.png"
        Image.new("RGB", (w, h)).save(p)
        paths.append(p)
    return paths


def test_crop_geometry_suggestions(tmp_path):
    from hydra_suite.training.pose_geometry_measure import (
        measure_crop_geometry,
        suggest_vitpose_hw,
        suggest_yolo_imgsz,
    )

    paths = _write_images(tmp_path / "imgs", [(192, 128)] * 5)
    (tmp_path / "imgs" / "broken.png").write_bytes(b"not an image")
    stats = measure_crop_geometry(paths + [tmp_path / "imgs" / "broken.png"])
    assert stats.images_measured == 5 and stats.images_skipped == 1
    assert stats.yolo_imgsz == 192
    # 192x128 crop, 1.25 box padding -> 240x160 -> grid of 32
    assert stats.vitpose_hw == [160, 256]
    assert suggest_yolo_imgsz(1000) == 640  # capped
    assert suggest_yolo_imgsz(20) == 64  # floor
    hw, clamped = suggest_vitpose_hw(192, 128, detail=2.0)
    assert clamped and max(hw) == 384
    with pytest.raises(ValueError):
        measure_crop_geometry([tmp_path / "imgs" / "broken.png"])


def test_box_padding_copy_matches_vitpose():
    from hydra_suite.core.individual.pose.vitpose.config import PADDING_FACTOR
    from hydra_suite.training.pose_geometry_measure import VITPOSE_BOX_PADDING

    assert VITPOSE_BOX_PADDING == PADDING_FACTOR


# ── YOLO dataset ─────────────────────────────────────────────────────────


def _labelled_crops(tmp_path: Path, n: int = 4):
    imgs = _write_images(tmp_path / "images", [(64, 48)] * n)
    labels = tmp_path / "labels"
    labels.mkdir()
    for p in imgs:
        kpts = " ".join(f"{0.2 + 0.1 * j:.3f} {0.4:.3f} 2" for j in range(len(ANT)))
        (labels / f"{p.stem}.txt").write_text(f"0 0.5 0.5 0.8 0.6 {kpts}\n")
    return imgs, labels


def test_yolo_dataset_yaml_carries_flip_idx(tmp_path):
    import yaml

    from hydra_suite.posekit.core.extensions import build_yolo_pose_dataset

    imgs, labels = _labelled_crops(tmp_path)
    info = build_yolo_pose_dataset(
        imgs,
        labels,
        tmp_path / "ds",
        0.75,
        0,
        ["ant"],
        ANT,
        flip_idx=[1, 0, 3, 2, 4, 5],
    )
    data = yaml.safe_load(Path(info["yaml_path"]).read_text())
    assert data["flip_idx"] == [1, 0, 3, 2, 4, 5]

    info = build_yolo_pose_dataset(
        imgs, labels, tmp_path / "ds2", 0.75, 0, ["ant"], ANT
    )
    assert "flip_idx" not in yaml.safe_load(Path(info["yaml_path"]).read_text())

    with pytest.raises(ValueError):
        build_yolo_pose_dataset(
            imgs, labels, tmp_path / "ds3", 0.75, 0, ["ant"], ANT, flip_idx=[0, 0, 1]
        )


def test_scale_balance_ignores_posekit_list_manifest(tmp_path):
    from hydra_suite.training.ultralytics_scale_balance import (
        _balance_settings_from_argv,
    )

    (tmp_path / "dataset.yaml").write_text("path: .\n")
    (tmp_path / "manifest.json").write_text(json.dumps([{"src_image": "a.png"}]))
    assert _balance_settings_from_argv([f"data={tmp_path / 'dataset.yaml'}"]) is None


# ── ViTPose trainer ──────────────────────────────────────────────────────


def _coco_dataset(tmp_path: Path, file_name: str):
    import cv2

    ds = tmp_path / "coco"
    (ds / "images").mkdir(parents=True)
    img = np.zeros((40, 60, 3), np.uint8)
    img[:, :10] = 255  # bright left edge: shows which way the image flipped
    cv2.imwrite(str(ds / "images" / "a.png"), img)
    kps = [10.0, 20.0, 2, 50.0, 20.0, 2, 30.0, 5.0, 2]  # left, right, centre-top
    coco = {
        "images": [{"id": 1, "file_name": file_name, "width": 60, "height": 40}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "keypoints": kps,
                "num_keypoints": 3,
                "bbox": [10, 5, 40, 15],
            }
        ],
        "categories": [{"id": 1, "name": "ant"}],
    }
    (ds / "annotations.json").write_text(json.dumps(coco))
    return ds


def test_vitpose_image_path_accepts_both_layouts(tmp_path):
    pytest.importorskip("torch")
    from hydra_suite.core.individual.pose.vitpose.training.dataset import image_path

    assert image_path(tmp_path, "images/a.png") == tmp_path / "images" / "a.png"
    assert image_path(tmp_path, "a.png") == tmp_path / "images" / "a.png"


@pytest.mark.parametrize("file_name", ["images/a.png", "a.png"])
def test_vitpose_dataset_mirror_swaps_pairs(tmp_path, monkeypatch, file_name):
    pytest.importorskip("torch")
    from hydra_suite.core.individual.pose.vitpose.training import dataset as D

    ds_dir = _coco_dataset(tmp_path, file_name)
    seen = {}

    def capture(img, kp, flip_idx, axis):
        out_img, out_kp = real_mirror(img, kp, flip_idx, axis)
        seen.setdefault("calls", []).append((axis, img, out_img, kp, out_kp))
        return out_img, out_kp

    real_mirror = D._mirror
    monkeypatch.setattr(D, "_mirror", capture)
    ds = D.CocoKeypointsDataset(
        ds_dir, [1], sigma=2.0, augment=True, flip_pairs=[[0, 1]], fliplr=1.0
    )
    ds[0]
    axis, img, out_img, kp, out_kp = seen["calls"][0]
    assert axis == 1
    assert out_img[:, -10:].mean() > 200  # bright edge moved to the right
    # left point (x=10) mirrors to x=50 and is still called "left" (row 0)
    np.testing.assert_allclose(out_kp[0, :2], [60 - 50.0, 20.0])
    np.testing.assert_allclose(out_kp[1, :2], [60 - 10.0, 20.0])
    np.testing.assert_allclose(out_kp[2, :2], [60 - 30.0, 5.0])

    seen.clear()
    ud = D.CocoKeypointsDataset(
        ds_dir, [1], sigma=2.0, augment=True, flip_pairs=[[0, 1]], flipud=1.0
    )
    ud[0]
    axis, _, _, kp, out_kp = seen["calls"][0]
    assert axis == 0
    np.testing.assert_allclose(out_kp[2, :2], [30.0, 40 - 5.0])
    np.testing.assert_allclose(out_kp[0, :2], [50.0, 20.0])  # swapped partner

    seen.clear()
    val = D.CocoKeypointsDataset(
        ds_dir, [1], sigma=2.0, augment=False, flip_pairs=[[0, 1]], fliplr=1.0
    )
    val[0]
    assert not seen  # validation never flips


def test_run_config_flip_validation():
    pytest.importorskip("torch")
    from hydra_suite.core.individual.pose.vitpose.training.config import (
        validate_run_config,
    )

    base = dict(
        init_checkpoint="x.pth",
        variant="B",
        num_keypoints=4,
        dataset_dir="d",
        output_dir="o",
    )
    cfg = validate_run_config(dict(base, flip_pairs=[[0, 1]], fliplr=0.5, flipud=0.5))
    assert cfg.flip_pairs == [[0, 1]] and cfg.fliplr == 0.5
    for bad in (
        {"fliplr": 1.5},
        {"flip_pairs": [[0, 0]]},
        {"flip_pairs": [[0, 1], [1, 2]]},
        {"flip_pairs": [[0, 9]]},
    ):
        with pytest.raises(ValueError):
            validate_run_config(dict(base, **bad))
    assert validate_run_config(dict(base)).fliplr == 0.0  # old run.json still loads


def test_ap10k_is_the_default_checkpoint():
    from hydra_suite.posekit.core.vitpose_checkpoints import (
        CATALOG,
        check_variant_available,
    )

    first = next(iter(CATALOG))
    assert first == "vitpose-b-ap10k"
    entry = CATALOG[first]
    assert entry.variant == "B" and len(entry.sha256) == 64
    check_variant_available("b")


# ── project + skeleton editor ────────────────────────────────────────────


def _project(tmp_path: Path, **kw):
    from hydra_suite.posekit.gui.models import Project

    img = tmp_path / "images"
    img.mkdir(exist_ok=True)
    lab = tmp_path / "labels"
    lab.mkdir(exist_ok=True)
    return Project(
        images_dir=img,
        out_root=tmp_path,
        labels_dir=lab,
        project_path=tmp_path / "pose_project.json",
        class_names=["ant"],
        keypoint_names=list(ANT),
        skeleton_edges=[(0, 2)],
        **kw,
    )


def test_project_round_trips_flip_pairs(tmp_path):
    from hydra_suite.posekit.gui.models import Project

    proj = _project(tmp_path, flip_pairs=[(0, 1), (2, 3)])
    proj.project_path.write_text(json.dumps(proj.to_json()))
    assert Project.from_json(proj.project_path).flip_pairs == [(0, 1), (2, 3)]

    data = proj.to_json()
    data["flip_pairs"] = [[0, 1], [1, 2], [4, 99]]  # reuse + out of range
    proj.project_path.write_text(json.dumps(data))
    assert Project.from_json(proj.project_path).flip_pairs == [(0, 1)]

    del data["flip_pairs"]  # projects saved before this change
    proj.project_path.write_text(json.dumps(data))
    assert Project.from_json(proj.project_path).flip_pairs == []


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    import hydra_suite.posekit.gui.utils as gu

    path = tmp_path / "ui_settings.json"
    monkeypatch.setattr(gu, "get_ui_settings_path", lambda: path)
    return path


def test_skeleton_editor_pairs_follow_edits_and_files(qapp, tmp_path, monkeypatch):
    from hydra_suite.posekit.gui.dialogs.skeleton import SkeletonEditorDialog

    dlg = SkeletonEditorDialog(list(ANT), [(0, 2)], None, flip_pairs=[])
    dlg.flip_panel._suggest()
    assert dlg.get_flip_pairs() == [(0, 1), (2, 3)]

    dlg.kpt_table.selectRow(0)
    dlg._kp_move(1)  # left_antenna_tip moves below right_antenna_tip
    names, _ = dlg.get_result()
    assert names[:2] == ["right_antenna_tip", "left_antenna_tip"]
    assert dlg.get_flip_pairs() == [(0, 1), (2, 3)]

    dlg.kpt_table.selectRow(2)
    dlg._kp_del()
    assert dlg.get_flip_pairs() == [(0, 1)]

    out = tmp_path / "skel.json"
    monkeypatch.setattr(dlg, "_pick_skeleton_path", lambda save: out)
    import hydra_suite.posekit.gui.dialogs.skeleton as S

    def _no_modal(*_a, **_k):
        raise AssertionError("unexpected message box")

    monkeypatch.setattr(S.QMessageBox, "critical", _no_modal)
    monkeypatch.setattr(S.QMessageBox, "question", _no_modal)
    dlg._save_config()
    saved = json.loads(out.read_text())
    assert saved["flip_pairs"] == [[0, 1]]

    # a file without flip_pairs keeps the current pairs
    old = {"keypoint_names": dlg.get_result()[0], "skeleton_edges": [[0, 2]]}
    out.write_text(json.dumps(old))
    dlg._load_config()
    assert dlg.get_flip_pairs() == [(0, 1)]
    saved["flip_pairs"] = []
    saved["skeleton_edges"] = [[0, 2]]
    out.write_text(json.dumps(saved))
    dlg._load_config()
    assert dlg.get_flip_pairs() == []


def test_bundled_ant_skeleton_has_pairs():
    from importlib import resources

    data = json.loads(
        resources.files("hydra_suite.resources")
        .joinpath("configs/skeletons/ooceraea_biroi.json")
        .read_text()
    )
    names = data["keypoint_names"]
    for a, b in data["flip_pairs"]:
        assert names[a].replace("left", "right") == names[b]


# ── Training Runner ──────────────────────────────────────────────────────


@pytest.fixture
def runner(qapp, tmp_path, isolated_settings):
    from hydra_suite.posekit.gui.dialogs.training import TrainingRunnerDialog

    proj = _project(tmp_path)
    imgs = _write_images(proj.images_dir, [(192, 128)] * 6)
    proj.project_path.write_text(json.dumps(proj.to_json()))
    dlg = TrainingRunnerDialog(None, proj, imgs)
    yield dlg
    dlg._loss_timer.stop()
    dlg.deleteLater()


def test_runner_defaults(runner):
    d = runner
    assert d.backend_combo.currentText() == "ViTPose"
    assert d.vitpose_checkpoint_combo.currentText() == "vitpose-b-ap10k"
    assert not d.cb_mosaic.isChecked()
    assert d.imgsz_spin.value() == 192 and not d.imgsz_spin.isEnabled()
    assert (d.vitpose_h_spin.value(), d.vitpose_w_spin.value()) == (160, 256)
    # ViTPose reads epochs / batch / device from Config, so it stays visible
    assert d.cfg_group.isVisibleTo(d)
    assert d.cfg_layout.isRowVisible(d.epochs_spin)
    assert not d.cfg_layout.isRowVisible(d._imgsz_row)
    assert "Flips off" in d.lbl_vitpose_flip_note.text()


def test_runner_manual_sizes_and_settings_round_trip(runner, isolated_settings):
    d = runner
    d.imgsz_auto_cb.setChecked(False)
    assert d.imgsz_spin.isEnabled()
    d.imgsz_spin.setValue(320)
    d.cb_mosaic.setChecked(True)
    d.fliplr_spin.setValue(0.3)
    d._save_settings()
    saved = json.loads(isolated_settings.read_text())["dialogs"]["training_runner"]
    assert saved["imgsz_auto"] is False and saved["imgsz"] == 320
    assert saved["mosaic"] is True and saved["fliplr"] == pytest.approx(0.3)

    from hydra_suite.posekit.gui.dialogs.training import TrainingRunnerDialog

    again = TrainingRunnerDialog(None, d.project, d.image_paths)
    assert again.imgsz_spin.value() == 320 and again.cb_mosaic.isChecked()
    again._loss_timer.stop()


def test_yolo_worker_passes_mosaic_and_flips(monkeypatch, tmp_path):
    import hydra_suite.posekit.gui.dialogs.training as T

    cmds = []

    class _Proc:
        stdout = iter(())

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(
        T.subprocess, "Popen", lambda cmd, **kw: cmds.append(cmd) or _Proc()
    )

    def run(augment):
        w = T.TrainingWorker(
            "m.pt",
            tmp_path / "d.yaml",
            tmp_path / "run",
            1,
            1,
            2,
            192,
            "cpu",
            augment,
            False,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            mosaic=1.0,
            fliplr=0.5,
            flipud=0.25,
        )
        w.run()
        return cmds[-1]

    on = run(True)
    assert "mosaic=1.0" in on and "fliplr=0.5" in on and "flipud=0.25" in on
    off = run(False)  # augmentations off also switches Ultralytics' defaults off
    assert "mosaic=0.0" in off and "fliplr=0.0" in off and "flipud=0.0" in off


def test_vitpose_worker_passes_flips(monkeypatch, tmp_path):
    import hydra_suite.posekit.gui.dialogs.training as T

    captured = {}
    monkeypatch.setattr(
        T,
        "build_coco_keypoints_dataset",
        lambda **kw: {"dataset_dir": tmp_path / "ds"},
    )
    monkeypatch.setattr(
        T,
        "prepare_run",
        lambda params, run_dir, cache_dir: captured.update(params)
        or Path(run_dir) / "run.json",
    )
    monkeypatch.setattr(T, "build_training_command", lambda rj: ["true"])

    class _Proc:
        stdout = iter(())

        def wait(self):
            return 0

    monkeypatch.setattr(T.subprocess, "Popen", lambda *a, **k: _Proc())
    w = T.ViTPoseTrainingWorker(
        image_paths=[],
        labels_dir=tmp_path,
        run_dir=tmp_path / "run",
        cache_dir=tmp_path / "c",
        class_names=["a"],
        keypoint_names=["l", "r"],
        skeleton_edges=[],
        variant="B",
        init_checkpoint="vitpose-b-ap10k",
        num_keypoints=2,
        epochs=1,
        batch=1,
        device="cpu",
        flip_pairs=[(0, 1)],
        fliplr=0.5,
        flipud=0.5,
    )
    w.run()
    assert captured["flip_pairs"] == [[0, 1]]
    assert captured["fliplr"] == 0.5 and captured["flipud"] == 0.5


# ── main window ──────────────────────────────────────────────────────────


def test_saved_prediction_backend_needs_a_model():
    from hydra_suite.posekit.gui.main_window import MainWindow

    has = MainWindow._saved_backend_has_model
    assert not has("YOLO", {"pred_backend": "YOLO", "pred_weights": ""})
    assert has("YOLO", {"pred_weights": "/m/best.pt"})
    assert has("ViTPose", {"pred_vitpose": "/v/best.pt"})
    assert not has("SLEAP", {})
    assert not has("other", {"pred_weights": "x"})
