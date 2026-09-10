"""Every path-bearing engine param key is classified, and only live roles ship."""

import pytest

from hydra_suite.trackerkit.engine_params import (
    MODEL_DIR_PARAM_KEYS,
    MODEL_FILE_PARAM_KEYS,
    MODEL_LIST_PARAM_KEYS,
    NON_MODEL_PATH_PARAM_KEYS,
    RuntimeContext,
    build_engine_params,
    iter_model_references,
)


def _runtime():
    return RuntimeContext(fps=30.0, total_frames=10, frame_width=64, frame_height=64)


def _everything_on(models_root):
    """A config that turns on every model-consuming role at once."""
    return {
        "detection_method": "yolo_obb",
        "yolo_obb_mode": "direct",
        "yolo_obb_direct_model_path": "obb/direct.pt",
        "yolo_detect_model_path": "detection/detect.pt",
        "yolo_crop_obb_model_path": "obb/cropped/crop.pt",
        "enable_headtail_orientation": True,
        "yolo_headtail_model_path": "classification/orientation/ht.pth",
        "enable_pose_extractor": True,
        "pose_model_type": "SLEAP",
        "pose_sleap_model_dir": "pose/SLEAP/run",
        "pose_model_dir": "pose/SLEAP/run",
        "pose_skeleton_file": str(models_root / "skel.json"),
        "enable_identity_analysis": True,
        "identity_method": "cnn",
        "cnn_classifiers": [{"model_path": "classification/identity/ids.pth"}],
        "color_tag_model_path": "classification/colortag/tags.pth",
        "use_apriltags": False,
    }


@pytest.fixture()
def models_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    for rel in (
        "obb/direct.pt",
        "detection/detect.pt",
        "obb/cropped/crop.pt",
        "classification/orientation/ht.pth",
        "classification/identity/ids.pth",
        "classification/colortag/tags.pth",
    ):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"m")
    (root / "pose" / "SLEAP" / "run").mkdir(parents=True)
    (root / "pose" / "SLEAP" / "run" / "best.ckpt").write_bytes(b"c")
    (root / "skel.json").write_text("{}")
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(root))
    return root


def test_every_path_ish_key_is_classified(models_root):
    """A new model role added without classification fails HERE, loudly.

    Minor fix: `path_ish` used to be computed from a SINGLE direct-mode,
    non-apriltag `_everything_on` build. `path_ish` only ever contains keys
    that are actually PRESENT in that one `params` dict -- a new
    path-bearing key that only appears when `yolo_obb_mode == "sequential"`
    or `use_apriltags == True` (e.g. a hypothetical
    APRILTAG_CALIBRATION_PATH) would never show up in `path_ish` at all
    under the direct-mode-only build, so the guard would stay silently
    green even if that key were never classified -- the exact
    "new role added without classification" failure this test exists to
    catch. Union `path_ish` across THREE builds: the direct-mode one above,
    a sequential-mode variant, and a `use_apriltags=True` variant, so a key
    that only exists under either of those modes is still covered."""
    cfg_direct = _everything_on(models_root)
    cfg_sequential = _everything_on(models_root)
    cfg_sequential["yolo_obb_mode"] = "sequential"
    cfg_apriltags = _everything_on(models_root)
    cfg_apriltags["use_apriltags"] = True

    path_ish: set[str] = set()
    for cfg in (cfg_direct, cfg_sequential, cfg_apriltags):
        params = build_engine_params(cfg, runtime=_runtime())
        path_ish |= {
            key
            for key in params
            if key.endswith("_PATH") or key.endswith("_DIR") or key.endswith("_FILE")
        }

    classified = (
        set(MODEL_FILE_PARAM_KEYS)
        | set(MODEL_DIR_PARAM_KEYS)
        | set(MODEL_LIST_PARAM_KEYS)
        | set(NON_MODEL_PATH_PARAM_KEYS)
    )
    unclassified = path_ish - classified
    assert not unclassified, (
        "New path-bearing engine param key(s) are unclassified: "
        f"{sorted(unclassified)}. Add each to exactly one of "
        "MODEL_FILE_PARAM_KEYS / MODEL_DIR_PARAM_KEYS / MODEL_LIST_PARAM_KEYS / "
        "NON_MODEL_PATH_PARAM_KEYS in engine_params.py."
    )


def test_list_carriers_are_classified(models_root):
    """Any list-of-dicts param containing a 'model_path' must be declared."""
    params = build_engine_params(_everything_on(models_root), runtime=_runtime())
    carriers = {
        key
        for key, value in params.items()
        if isinstance(value, list)
        and value
        and all(isinstance(e, dict) for e in value)
        and any("model_path" in e for e in value)
    }
    assert carriers <= set(
        MODEL_LIST_PARAM_KEYS
    ), f"Undeclared model-carrying list param(s): {sorted(carriers - set(MODEL_LIST_PARAM_KEYS))}"


def test_classification_sets_are_disjoint():
    sets = [
        set(MODEL_FILE_PARAM_KEYS),
        set(MODEL_DIR_PARAM_KEYS),
        set(MODEL_LIST_PARAM_KEYS),
        set(NON_MODEL_PATH_PARAM_KEYS),
    ]
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            assert not (a & b), f"key classified twice: {sorted(a & b)}"


def test_all_live_roles_are_yielded(models_root):
    params = build_engine_params(_everything_on(models_root), runtime=_runtime())
    roles = {ref.role for ref in iter_model_references(params)}
    # COLOR_TAG_MODEL_PATH is intentionally absent: it is dead (no consumer in
    # src/hydra_suite/core/; the GUI field is setVisible(False)) and moved to
    # NON_MODEL_PATH_PARAM_KEYS below. It is never yielded regardless of value.
    assert roles == {
        "YOLO_OBB_DIRECT_MODEL_PATH",
        "YOLO_HEADTAIL_MODEL_PATH",
        "POSE_MODEL_DIR",
        "CNN_CLASSIFIERS",
    }


def test_pose_is_not_shipped_when_the_stage_is_off(models_root):
    cfg = _everything_on(models_root)
    cfg["enable_pose_extractor"] = False
    params = build_engine_params(cfg, runtime=_runtime())
    assert params["POSE_MODEL_DIR"], "precondition: the key is still emitted"
    roles = {ref.role for ref in iter_model_references(params)}
    assert "POSE_MODEL_DIR" not in roles


def test_unselected_yolo_mode_models_are_not_shipped(models_root):
    """In direct mode the sequential pair is emitted but never loaded."""
    params = build_engine_params(_everything_on(models_root), runtime=_runtime())
    assert params["YOLO_DETECT_MODEL_PATH"], "precondition: emitted anyway"
    roles = {ref.role for ref in iter_model_references(params)}
    assert "YOLO_DETECT_MODEL_PATH" not in roles
    assert "YOLO_CROP_OBB_MODEL_PATH" not in roles


def test_sequential_mode_ships_the_pair_and_not_the_direct_model(models_root):
    cfg = _everything_on(models_root)
    cfg["yolo_obb_mode"] = "sequential"
    params = build_engine_params(cfg, runtime=_runtime())
    roles = {ref.role for ref in iter_model_references(params)}
    assert "YOLO_DETECT_MODEL_PATH" in roles
    assert "YOLO_CROP_OBB_MODEL_PATH" in roles
    assert "YOLO_OBB_DIRECT_MODEL_PATH" not in roles


def test_cnn_classifiers_ship_even_when_identity_flag_is_off(models_root):
    """Core does not gate CNN classifiers on ENABLE_IDENTITY_ANALYSIS — neither
    does iter_model_references. See core/inference/config.py:1224,
    core/inference/runner.py:509, core/tracking/worker.py:955: the flag is
    never consulted for whether classifiers load. A flag-gated
    iter_model_references would under-ship relative to what core actually
    loads and cause silent divergence on the remote box."""
    cfg = _everything_on(models_root)
    cfg["enable_identity_analysis"] = False
    params = build_engine_params(cfg, runtime=_runtime())
    roles = {ref.role for ref in iter_model_references(params)}
    assert "CNN_CLASSIFIERS" in roles


def test_empty_cnn_classifiers_list_yields_nothing(models_root):
    cfg = _everything_on(models_root)
    cfg["cnn_classifiers"] = []
    params = build_engine_params(cfg, runtime=_runtime())
    roles = {ref.role for ref in iter_model_references(params)}
    assert "CNN_CLASSIFIERS" not in roles


def test_color_tag_model_path_is_never_yielded(models_root):
    """COLOR_TAG_MODEL_PATH is dead (no consumer in core/); it lives in
    NON_MODEL_PATH_PARAM_KEYS and must never appear as a reference role even
    when populated and even when identity is enabled."""
    params = build_engine_params(_everything_on(models_root), runtime=_runtime())
    assert params["COLOR_TAG_MODEL_PATH"], "precondition: the key is populated"
    roles = {ref.role for ref in iter_model_references(params)}
    assert "COLOR_TAG_MODEL_PATH" not in roles


def test_stale_color_tag_path_does_not_block_yielding_other_roles(models_root):
    """A nonexistent color_tag_model_path must not affect iter_model_references
    at all, since the key is never resolved to a filesystem check here (that
    dead-key handling lives entirely in NON_MODEL_PATH_PARAM_KEYS)."""
    cfg = _everything_on(models_root)
    cfg["color_tag_model_path"] = "classification/colortag/does_not_exist.pth"
    params = build_engine_params(cfg, runtime=_runtime())
    roles = {ref.role for ref in iter_model_references(params)}
    assert "CNN_CLASSIFIERS" in roles
    assert "YOLO_OBB_DIRECT_MODEL_PATH" in roles


def test_pose_reference_kind_is_directory(models_root):
    params = build_engine_params(_everything_on(models_root), runtime=_runtime())
    pose = [r for r in iter_model_references(params) if r.role == "POSE_MODEL_DIR"]
    assert len(pose) == 1
    assert pose[0].kind == "directory"


def test_pose_reference_kind_is_file_for_yolo_pose(models_root):
    """Fix A3: POSE_MODEL_DIR is a FILE for the YOLO-pose/ViTPose backends —
    pose/backends/yolo.py:64-65 and vitpose.py:206 both call
    model_path.with_suffix(...) on it. "kind" must be derived from what's
    on disk, not assumed directory just because the key lives in
    MODEL_DIR_PARAM_KEYS. **Correction (fix V5): the real
    ant_pose_headtail.json fixture does NOT exercise this file-kind branch —
    it has `"pose_model_type": "sleap"` (verified:
    tools/equivalence/fixtures/configs/ant_pose_headtail.json:236), so
    POSE_MODEL_DIR there resolves to the SLEAP run DIRECTORY, not a file.
    This test therefore builds its own synthetic YOLO-pose config from
    scratch (below) rather than citing the fixture — that part was always
    correct — but a prior draft's rationale wrongly implied the fixture
    itself was the YOLO-pose file-kind case. It is not: it is Task 13's
    SLEAP/directory case (see the Task 13 pre-check below for what that
    implies for the Goal-4 portability probe)."""
    cfg = _everything_on(models_root)
    yolo_pose_path = models_root / "pose" / "YOLO-pose" / "run.pt"
    yolo_pose_path.parent.mkdir(parents=True, exist_ok=True)
    yolo_pose_path.write_bytes(b"y")
    cfg["pose_model_type"] = "YOLO"
    cfg["pose_yolo_model_dir"] = "pose/YOLO-pose/run.pt"
    cfg["pose_model_dir"] = "pose/YOLO-pose/run.pt"
    params = build_engine_params(cfg, runtime=_runtime())
    pose = [r for r in iter_model_references(params) if r.role == "POSE_MODEL_DIR"]
    assert len(pose) == 1
    assert pose[0].kind == "file"


def test_empty_values_are_skipped(models_root):
    cfg = _everything_on(models_root)
    cfg["enable_headtail_orientation"] = False
    params = build_engine_params(cfg, runtime=_runtime())
    assert params["YOLO_HEADTAIL_MODEL_PATH"] == ""
    assert "YOLO_HEADTAIL_MODEL_PATH" not in {
        r.role for r in iter_model_references(params)
    }


def test_bgsub_config_yields_no_model_references(models_root):
    cfg = _everything_on(models_root)
    cfg["detection_method"] = "background_subtraction"
    cfg["enable_pose_extractor"] = False
    cfg["enable_identity_analysis"] = False
    cfg["enable_headtail_orientation"] = False
    # Fix B3: clearing the LIST is what makes this assertion true, not clearing
    # the enable flag. The C1 rule (below) is that CNN_CLASSIFIERS is yielded
    # whenever the list is non-empty, because core/inference/config.py:1224
    # builds a CNNConfig per entry with no reference to
    # ENABLE_IDENTITY_ANALYSIS. A bgsub job that still ships classifiers is
    # therefore correct behaviour, not a bug -- so this test must empty the
    # list to assert "no model references at all".
    cfg["cnn_classifiers"] = []
    params = build_engine_params(cfg, runtime=_runtime())
    assert list(iter_model_references(params)) == []
