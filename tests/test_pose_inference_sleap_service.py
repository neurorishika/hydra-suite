from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.module_loader import load_src_module


def _load_pose_inference_module():
    return load_src_module(
        "hydra_suite/integrations/sleap/service.py",
        "pose_inference_sleap_service_under_test",
    )


def test_all_zero_pose_chunk_detection(tmp_path: Path) -> None:
    mod = _load_pose_inference_module()
    frames = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    zero_preds = {
        str(frames[0]): [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        str(frames[1]): [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    }
    assert mod._all_zero_pose_chunk(frames, zero_preds, num_keypoints=2) is True

    mixed_preds = {
        str(frames[0]): [(0.0, 0.0, 0.0), (1.0, 2.0, 0.4)],
        str(frames[1]): [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    }
    assert mod._all_zero_pose_chunk(frames, mixed_preds, num_keypoints=2) is False


def test_sleap_service_all_zero_onnx_triggers_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_pose_inference_module()

    class _FakeService:
        def __init__(self):
            self.log_path = None

        def start(self, _env_name: str, log_path=None):
            self.log_path = log_path
            return True, ""

        def request(self, _path: str, payload: dict, timeout: float = 3600.0):
            _ = timeout
            preds = {
                str(Path(p)): [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
                for p in payload.get("images", [])
            }
            return {"ok": True, "preds": preds}

    fake_service = _FakeService()
    monkeypatch.setattr(mod, "_get_sleap_service", lambda: fake_service)

    fallback_called = {"n": 0}

    def _fake_fallback(**kwargs):
        fallback_called["n"] += 1
        images = kwargs.get("image_paths", [])
        preds = {str(Path(p)): [(10.0, 20.0, 0.8), (30.0, 40.0, 0.7)] for p in images}
        return True, preds, ""

    monkeypatch.setattr(mod, "_run_sleap_export_predict_subprocess", _fake_fallback)

    out_json = tmp_path / "posekit" / "tmp" / "preds.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    images = [tmp_path / "f1.jpg", tmp_path / "f2.jpg"]
    ok, preds, err = mod._run_sleap_predict_service(
        model_dir=tmp_path / "sleap_model",
        image_paths=images,
        out_json=out_json,
        keypoint_names=["head", "tail"],
        skeleton_edges=[],
        env_name="sleap",
        device="cpu",
        batch=2,
        max_instances=1,
        runtime_flavor="onnx",
        exported_model_path=str(tmp_path / "m.onnx"),
        export_input_hw=(224, 224),
    )

    assert ok is True
    assert err == ""
    assert fallback_called["n"] == 1
    assert preds[str(images[0])][0] == (10.0, 20.0, 0.8)
    assert preds[str(images[1])][1] == (30.0, 40.0, 0.7)


def test_sleap_service_nonzero_onnx_does_not_trigger_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_pose_inference_module()

    class _FakeService:
        log_path = None

        def start(self, _env_name: str, log_path=None):
            self.log_path = log_path
            return True, ""

        def request(self, _path: str, payload: dict, timeout: float = 3600.0):
            _ = timeout
            preds = {
                str(Path(p)): [(11.0, 22.0, 0.6), (33.0, 44.0, 0.5)]
                for p in payload.get("images", [])
            }
            return {"ok": True, "preds": preds}

    monkeypatch.setattr(mod, "_get_sleap_service", lambda: _FakeService())
    fallback_called = {"n": 0}

    def _fake_fallback(**kwargs):
        _ = kwargs
        fallback_called["n"] += 1
        return True, {}, ""

    monkeypatch.setattr(mod, "_run_sleap_export_predict_subprocess", _fake_fallback)

    out_json = tmp_path / "posekit" / "tmp" / "preds.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    images = [tmp_path / "f1.jpg"]
    ok, preds, err = mod._run_sleap_predict_service(
        model_dir=tmp_path / "sleap_model",
        image_paths=images,
        out_json=out_json,
        keypoint_names=["head", "tail"],
        skeleton_edges=[],
        env_name="sleap",
        runtime_flavor="onnx",
        exported_model_path=str(tmp_path / "m.onnx"),
    )

    assert ok is True
    assert err == ""
    assert fallback_called["n"] == 0
    assert preds[str(images[0])][0] == (11.0, 22.0, 0.6)


def test_sleap_service_collects_chunk_timing_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_pose_inference_module()

    class _FakeService:
        log_path = None

        def start(self, _env_name: str, log_path=None):
            self.log_path = log_path
            return True, ""

        def request(self, _path: str, payload: dict, timeout: float = 3600.0):
            _ = timeout
            preds = {
                str(Path(p)): [(1.0, 2.0, 0.5), (3.0, 4.0, 0.4)]
                for p in payload.get("images", [])
            }
            return {
                "ok": True,
                "preds": preds,
                "timings_s": {
                    "service_decode_s": 0.01,
                    "service_inference_s": 0.02,
                },
            }

    monkeypatch.setattr(mod, "_get_sleap_service", lambda: _FakeService())

    out_json = tmp_path / "posekit" / "tmp" / "preds.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    images = [tmp_path / f"f{i}.jpg" for i in range(12)]
    metrics = {}
    ok, preds, err = mod._run_sleap_predict_service(
        model_dir=tmp_path / "sleap_model",
        image_paths=images,
        out_json=out_json,
        keypoint_names=["head", "tail"],
        skeleton_edges=[],
        env_name="sleap",
        batch=1,
        metrics_out=metrics,
    )

    assert ok is True
    assert err == ""
    assert len(preds) == len(images)
    assert metrics["service_decode_s"] == pytest.approx(0.02)
    assert metrics["service_inference_s"] == pytest.approx(0.04)


def test_format_sleap_env_preflight_error_torchvision_nms() -> None:
    mod = _load_pose_inference_module()
    msg = mod._format_sleap_env_preflight_error(
        "RuntimeError: operator torchvision::nms does not exist", "sleap"
    )
    assert "torchvision::nms" in msg
    assert "torch and torchvision are incompatible builds" in msg
    assert "conda run -n sleap" in msg


def test_format_sleap_env_preflight_error_nccl_symbol() -> None:
    mod = _load_pose_inference_module()
    msg = mod._format_sleap_env_preflight_error(
        "ERROR: torch import failed: /x/libtorch_cuda.so: undefined symbol: ncclAlltoAll",
        "sleap",
    )
    assert "CUDA/NCCL binary mismatch" in msg
    assert "nvidia-nccl-cu13" in msg
    assert "conda run -n sleap" in msg


def test_sleap_service_start_returns_preflight_failure(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_pose_inference_module()
    svc = mod._SleapHttpService()

    monkeypatch.setattr(mod.shutil, "which", lambda _name: "/usr/bin/conda")
    monkeypatch.setattr(
        mod,
        "_sleap_env_preflight",
        lambda _env_name: (False, "preflight import failure"),
    )

    ok, err = svc.start("sleap", log_path=tmp_path / "sleap_service.log")
    assert ok is False
    assert "preflight import failure" in err
    assert svc.proc is None


def test_sleap_env_preflight_runs_temp_script(tmp_path: Path, monkeypatch) -> None:
    mod = _load_pose_inference_module()

    class _Res:
        returncode = 0
        stdout = '{"ok": true}'

    captured = {"cmd": None, "script_path": None}

    def _fake_run(cmd, **kwargs):
        _ = kwargs
        captured["cmd"] = list(cmd)
        captured["script_path"] = Path(cmd[-1])
        return _Res()

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(mod.tempfile, "gettempdir", lambda: str(tmp_path))

    ok, err = mod._sleap_env_preflight("sleap")
    assert ok is True
    assert err == ""
    assert captured["cmd"] is not None
    assert "-c" not in captured["cmd"]
    assert captured["script_path"] is not None
    assert captured["script_path"].suffix == ".py"
    assert captured["script_path"].exists() is False


def test_sleap_service_code_normalizes_export_logits() -> None:
    mod = _load_pose_inference_module()
    code = str(mod._SLEAP_SERVICE_CODE)
    assert "def _normalize_conf_values(conf):" in code
    assert "1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0)))" in code


def test_sleap_service_code_uses_rgb_letterbox_for_exported_crops() -> None:
    mod = _load_pose_inference_module()
    code = str(mod._SLEAP_SERVICE_CODE)
    assert "arr = arr[:, :, ::-1].copy()" in code
    assert "scale = min(float(w) / float(orig_w), float(h) / float(orig_h))" in code
    assert "canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized" in code
    assert "arr[:, 0] = (arr[:, 0] - float(pad_x)) / scale" in code


def test_sleap_service_code_supports_in_memory_native_video_creation() -> None:
    mod = _load_pose_inference_module()
    code = str(mod._SLEAP_SERVICE_CODE)
    assert "tempfile,time,uuid" in code
    assert "def _make_video_from_arrays(image_arrays):" in code
    assert "video = _make_video(native_images, image_arrays=image_arrays)" in code
    # Pose golden rule: the in-memory video path is authoritative. The old
    # disk-materialization fallback must be gone (fail loud, never round-trip).
    assert "falling back to temporary image files" not in code
    assert "_materialize_image_arrays" not in code
    assert "native_array_video_supported" in code


def test_sleap_service_code_prefers_from_model_paths_predictor_factory() -> None:
    mod = _load_pose_inference_module()
    code = str(mod._SLEAP_SERVICE_CODE)
    assert "if hasattr(predictor_cls, 'from_model_paths'):" in code
    assert "pred = predictor_cls.from_model_paths(" in code
    assert "model_paths=[model_dir]," in code
    assert "device=device," in code
    assert "def _load_training_preprocess_config(model_dir):" in code
    assert "cfg_path = Path(str(model_dir)) / 'training_config.yaml'" in code
    assert "preprocess_config=preprocess_config" in code
    assert "queue_maxsize = max(1, min(int(batch), 8))" in code
    assert "pred.make_pipeline(labels, queue_maxsize=queue_maxsize)" in code
    assert "return fn(make_labels=True)" in code


def test_pose_inference_service_reports_native_array_video_capability(
    monkeypatch,
) -> None:
    mod = _load_pose_inference_module()

    class _FakeService:
        def health(self, timeout: float = 2.0):
            _ = timeout
            return {"ok": True, "native_array_video_supported": True}

    monkeypatch.setattr(mod, "_get_sleap_service", lambda: _FakeService())

    assert mod.PoseInferenceService.sleap_service_health()["ok"] is True
    assert mod.PoseInferenceService.sleap_native_array_video_supported() is True
