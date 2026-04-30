from __future__ import annotations

import sys
import types

import verify_cuda_runtime as verify


class _FakeDist:
    def __init__(self, name: str) -> None:
        self.metadata = {"Name": name}


def test_verify_tensorrt_import_rejects_mixed_cuda_wheel_families(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        verify.importlib_metadata,
        "distributions",
        lambda: [_FakeDist("tensorrt-cu12"), _FakeDist("tensorrt-cu13")],
    )

    status = verify._verify_tensorrt_import()

    captured = capsys.readouterr()
    assert status == 1
    assert "Mixed TensorRT CUDA wheel families detected" in captured.err


def test_verify_tensorrt_import_accepts_single_cuda_family(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        verify.importlib_metadata,
        "distributions",
        lambda: [_FakeDist("tensorrt-cu12")],
    )

    fake_trt = types.SimpleNamespace(
        __version__="10.16.1.11",
        Logger=type(
            "Logger",
            (),
            {
                "ERROR": 0,
                "__init__": lambda self, _level: None,
            },
        ),
        Builder=type(
            "Builder",
            (),
            {
                "__init__": lambda self, _logger: None,
            },
        ),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    status = verify._verify_tensorrt_import()

    captured = capsys.readouterr()
    assert status == 0
    assert "TensorRT builder initialized successfully" in captured.out
