"""Classifier TensorRT sessions are built in a child process first.

A TRT builder crash (CUDA 716 "misaligned address" on a ViT-B/16) poisoned the
tracking process's CUDA context, so the native fallback could not load either.
"""

from __future__ import annotations

import subprocess
import sys

from hydra_suite.core.individual.classification.trt_probe import (
    probe_trt_session_in_child,
)
from hydra_suite.runtime.resolver import ResolvedBackend


def test_probe_reports_child_success_and_failure():
    def ok_runner(cmd, **kw):
        assert cmd[:3] == [
            sys.executable,
            "-m",
            "hydra_suite.core.individual.classification.trt_probe",
        ]
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def crash_runner(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, -6, "", "CUDA error: misaligned address"
        )

    providers = [
        ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
        "CPUExecutionProvider",
    ]
    assert probe_trt_session_in_child("m.onnx", providers, runner=ok_runner).ok
    bad = probe_trt_session_in_child("m.onnx", providers, runner=crash_runner)
    assert not bad.ok and bad.returncode == -6 and "misaligned" in bad.detail


def test_probe_child_really_builds_a_cpu_session(tiny_flat_headtail):
    from hydra_suite.core.individual.classification.backend import ClassifierBackend

    backend = ClassifierBackend(
        str(tiny_flat_headtail), resolved=ResolvedBackend("torch", "cpu", False)
    )
    peer = backend._derive_onnx_peer()
    assert probe_trt_session_in_child(str(peer), ["CPUExecutionProvider"]).ok
    assert not probe_trt_session_in_child(
        str(peer) + ".missing", ["CPUExecutionProvider"]
    ).ok


def test_failed_probe_falls_back_to_native_without_touching_parent_session(
    tiny_flat_headtail, monkeypatch
):
    import onnxruntime as ort

    import hydra_suite.core.individual.classification.backend as backend_module
    import hydra_suite.core.individual.classification.trt_probe as probe_module
    from hydra_suite.core.individual.classification.backend import ClassifierBackend

    def _crash(model_path, providers, runner=None):
        return probe_module.TrtProbeResult(False, -6, "CUDA error: misaligned address")

    def _no_parent_session(*a, **k):
        raise AssertionError("parent must not build a TRT session after a failed probe")

    monkeypatch.setattr(probe_module, "probe_trt_session_in_child", _crash)
    monkeypatch.setattr(ort, "InferenceSession", _no_parent_session, raising=True)
    monkeypatch.setattr(
        backend_module,
        "execution_providers_for",
        lambda resolved, **kw: ["TensorrtExecutionProvider", "CPUExecutionProvider"],
        raising=False,
    )
    import hydra_suite.runtime.onnx_providers as op

    monkeypatch.setattr(
        op,
        "execution_providers_for",
        lambda resolved, **kw: ["TensorrtExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        ClassifierBackend,
        "_build_trt_providers_with_profiles",
        lambda self, p, prov: prov,
    )
    monkeypatch.setattr(
        backend_module,
        "_available_onnx_provider_names",
        lambda: {"TensorrtExecutionProvider"},
    )
    monkeypatch.setattr(backend_module, "_native_accelerator_available", lambda r: True)
    loaded = {}
    monkeypatch.setattr(
        backend_module._TinyLoader,
        "load",
        staticmethod(
            lambda path, device: loaded.setdefault("device", device) and object()
        ),
    )
    monkeypatch.setattr(
        ClassifierBackend, "_warmup_native_cuda_model", lambda self: None
    )

    backend = ClassifierBackend(
        str(tiny_flat_headtail), resolved=ResolvedBackend("tensorrt", "cuda", False)
    )
    backend._ensure_loaded()
    assert backend._active_execution_backend == "native"
    assert loaded["device"] == "cuda"


def test_fp16_probe_failure_retries_tensorrt_in_fp32(tiny_flat_headtail, monkeypatch):
    """TensorRT 10.16 cannot build ViT-B/16 in fp16 at batch > 1 (CUDA 716);
    the same graph builds in fp32, which is ~1.9x faster than native fp32."""
    import onnxruntime as ort

    import hydra_suite.core.individual.classification.trt_probe as probe_module
    import hydra_suite.runtime.onnx_providers as op
    from hydra_suite.core.individual.classification.backend import ClassifierBackend

    probes = []

    def _probe(model_path, providers, runner=None):
        opts = providers[0][1]
        probes.append(opts["trt_fp16_enable"])
        if opts["trt_fp16_enable"]:
            return probe_module.TrtProbeResult(
                False, 1, "CUDA error 716: misaligned address"
            )
        return probe_module.TrtProbeResult(True, 0, "")

    parent = {}

    class _Session:
        def __init__(self, path, providers):
            parent["fp16"] = providers[0][1]["trt_fp16_enable"]

    monkeypatch.setattr(probe_module, "probe_trt_session_in_child", _probe)
    monkeypatch.setattr(ort, "InferenceSession", _Session)
    monkeypatch.setattr(
        op,
        "execution_providers_for",
        lambda resolved, **kw: ["TensorrtExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        ClassifierBackend,
        "_build_trt_providers_with_profiles",
        lambda self, p, prov: [
            ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
            "CPUExecutionProvider",
        ],
    )
    monkeypatch.setattr(ClassifierBackend, "_warmup_ort_session_trt", lambda self: None)

    backend = ClassifierBackend(
        str(tiny_flat_headtail), resolved=ResolvedBackend("tensorrt", "cuda", False)
    )
    backend._load_onnx()
    assert probes == [True, False]
    assert parent["fp16"] is False
