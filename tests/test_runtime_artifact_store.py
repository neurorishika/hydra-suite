"""Contract tests for immutable TensorRT/CoreML runtime-artifact storage."""

from __future__ import annotations

import multiprocessing
import threading
import time
from pathlib import Path

from hydra_suite.core.inference.runtime_artifacts import (
    RuntimeArtifactIdentity,
    RuntimeArtifactStore,
)


def _process_build_once(root: str, source: str, counter: str, queue) -> None:
    """Worker used to prove the store's lock works across interpreter processes."""
    source_path = Path(source)
    identity = _identity(source_path)
    store = RuntimeArtifactStore(Path(root))

    def build(_private_source: Path, output: Path) -> None:
        with Path(counter).open("a", encoding="utf-8") as handle:
            handle.write("build\n")
        time.sleep(0.1)
        output.write_bytes(b"engine")

    queue.put(store.ensure(identity, build, source_path=source_path).built)


def _identity(source: Path, **overrides: object) -> RuntimeArtifactIdentity:
    values: dict[str, object] = {
        "source_path": source,
        "runtime": "tensorrt",
        "runtime_fingerprint": "trt=10.0|cuda=12.8",
        "imgsz": 640,
        "task": "obb",
        "precision": "fp16",
        "profile": {"min_batch": 1, "opt_batch": 4, "max_batch": 4},
    }
    values.update(overrides)
    return RuntimeArtifactIdentity.from_source(**values)


def test_runtime_artifact_identity_includes_exact_profile_and_not_tuning_policy(
    tmp_path,
):
    source = tmp_path / "model.pt"
    source.write_bytes(b"weights")
    baseline = _identity(source)

    assert baseline != _identity(source, imgsz=1024)
    assert baseline != _identity(source, task="detect")
    assert baseline != _identity(source, precision="fp32")
    assert baseline != _identity(source, runtime_fingerprint="trt=10.1|cuda=12.8")
    assert baseline != _identity(
        source, profile={"min_batch": 1, "opt_batch": 8, "max_batch": 8}
    )
    source.write_bytes(b"different-weights")
    assert baseline != _identity(source)

    # Performance/tuning policy is deliberately absent: engine lifetime is
    # independent of a coordinate-search policy revision.
    assert "tuning" not in baseline.as_dict()


def test_store_builds_in_private_directory_and_atomically_promotes(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"weights")
    store = RuntimeArtifactStore(tmp_path / "runtime-artifacts")
    identity = _identity(source)
    observed: dict[str, Path] = {}

    def build(private_source: Path, output: Path) -> None:
        observed["private_source"] = private_source
        observed["output"] = output
        assert private_source.read_bytes() == b"weights"
        assert not output.exists()
        output.write_bytes(b"engine")

    result = store.ensure(identity, build, source_path=source)

    assert result.built is True
    assert result.path.read_bytes() == b"engine"
    assert result.path.parent.name == identity.digest
    assert observed["private_source"].parent == observed["output"].parent
    assert observed["output"].parent != result.path.parent
    assert store.is_ready(identity)
    assert not list((store.root / ".tmp").iterdir())


def test_store_failure_never_leaks_partial_artifact(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"weights")
    store = RuntimeArtifactStore(tmp_path / "runtime-artifacts")
    identity = _identity(source)

    def fail_build(_private_source: Path, output: Path) -> None:
        output.write_bytes(b"partial")
        raise RuntimeError("export failed")

    try:
        store.ensure(identity, fail_build, source_path=source)
    except RuntimeError as exc:
        assert "export failed" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("failing exporter unexpectedly succeeded")

    assert not store.path_for(identity).exists()
    assert not store.is_ready(identity)


def test_store_single_flight_serializes_concurrent_builders(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"weights")
    store = RuntimeArtifactStore(tmp_path / "runtime-artifacts")
    identity = _identity(source)
    starts: list[int] = []
    results = []
    barrier = threading.Barrier(2)

    def build(_private_source: Path, output: Path) -> None:
        starts.append(threading.get_ident())
        time.sleep(0.05)
        output.write_bytes(b"engine")

    def worker() -> None:
        barrier.wait()
        results.append(store.ensure(identity, build, source_path=source))

    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)
    first.start()
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert len(starts) == 1
    assert len(results) == 2
    assert sorted(result.built for result in results) == [False, True]
    assert all(result.path == results[0].path for result in results)


def test_store_single_flight_is_process_safe(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"weights")
    counter = tmp_path / "build-count.txt"
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    args = (str(tmp_path / "runtime-artifacts"), str(source), str(counter), queue)
    first = context.Process(target=_process_build_once, args=args)
    second = context.Process(target=_process_build_once, args=args)
    first.start()
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert sorted([queue.get(timeout=2), queue.get(timeout=2)]) == [False, True]
    assert counter.read_text(encoding="utf-8").splitlines() == ["build"]
