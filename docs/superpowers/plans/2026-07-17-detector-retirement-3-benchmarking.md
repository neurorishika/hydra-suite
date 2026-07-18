# Detector Retirement — Plan 3: Benchmarking Replacement (Phase D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all in-UI benchmarking and the old per-model/per-runtime CLI benchmark, and replace them with one external CLI tool that runs the whole `InferenceRunner` pipeline from a config file at each of the three runtime tiers (cpu / gpu / gpu_fast), reporting end-to-end timing with an opt-in engine-build-time diagnostic.

**Architecture:** The runtime system is now three speed-ordered tiers resolved deterministically per platform (`resolver.py`), so "which runtime is fastest" is no longer a user question. The new tool takes a config file → `build_inference_config_from_params`/`InferenceConfig` → constructs an `InferenceRunner` per tier (overriding `runtime_tier`) → times `run_realtime`/`run_batch_pass` over synthetic or sampled frames → prints a per-tier table. It drives only the runner (no `YOLOOBBDetector`, no low-level `load_obb_executor`).

**Tech Stack:** Python 3, argparse, `InferenceRunner`, `runtime/resolver.py`, numpy/opencv, pytest.

## Global Constraints

- **Depends on Plan 1 (Foundation).** Uses `build_inference_config_from_params` / `InferenceConfig` and `InferenceRunner`.
- **Independent of Plan 2** (can run in parallel), but **must land before Plan 4** — Plan 4 deletes `core/detectors`, and `tests/test_benchmark_models.py` hard-depends on `YOLOOBBDetector` being importable; that test is deleted here.
- The new CLI drives `InferenceRunner` only. No `YOLOOBBDetector`, no direct `load_obb_executor` in the tool.
- Run `make format` before each commit.

---

## File Structure

- `src/hydra_suite/trackerkit/benchmarking.py` — **delete** (in-UI benchmarking module).
- `src/hydra_suite/trackerkit/gui/dialogs/benchmark_dialog.py` — **delete** (the dialog).
- `src/hydra_suite/trackerkit/gui/orchestrators/config.py` — **remove** benchmark wiring (imports L34-37; calls L3665, L3712, L3714; `_build_optimizer_detection_cache` is unrelated — leave it).
- `src/hydra_suite/trackerkit/gui/main_window.py` — **remove** `_open_benchmark_dialog` (L1085) + its menu/button trigger.
- `tools/benchmark_models.py` — **delete** (old per-model/per-runtime CLI, ×10 `YOLOOBBDetector`).
- `tools/benchmark_pipeline.py` — **new** tier-based whole-pipeline CLI.
- `tests/test_trackerkit_benchmarking.py` — **delete**.
- `tests/test_benchmark_models.py` — **delete**.
- `tests/test_benchmark_pipeline.py` — **new**.

---

### Task 1: Build the new tier-based pipeline benchmark CLI

**Files:**
- Create: `tools/benchmark_pipeline.py`
- Test: `tests/test_benchmark_pipeline.py`

**Interfaces:**
- Consumes: `build_inference_config_from_params(params) -> InferenceConfig` (Plan 1); `dataclasses.replace` to override `runtime_tier`; `InferenceRunner(cfg, cache_dir=..., video_path=...)`; `runtime/resolver.available_tiers`, `detect_platform`, `tier_label`.
- Produces: `run_pipeline_benchmark(config_params: dict, *, tiers: list[str], iterations: int, warmup: int, frame_size: tuple[int, int], compile_timing: bool) -> list[dict]` and a `main()` argparse entry.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark_pipeline.py
import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "benchmark_pipeline",
    Path(__file__).resolve().parents[1] / "tools" / "benchmark_pipeline.py",
)
bp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bp)


def test_run_pipeline_benchmark_times_each_tier(monkeypatch):
    built = []

    class _FakeRunner:
        def __init__(self, cfg, cache_dir=None, video_path=None, cache_only=False):
            built.append(cfg.runtime_tier)

        def run_realtime(self, frame, frame_idx=0, roi_mask=None):
            return object()

        def close(self):
            pass

    monkeypatch.setattr(bp, "InferenceRunner", _FakeRunner)
    monkeypatch.setattr(
        bp, "build_inference_config_from_params", lambda p: bp._Cfg(runtime_tier="cpu")
    )

    rows = bp.run_pipeline_benchmark(
        {"DETECTION_METHOD": "yolo_obb"},
        tiers=["cpu", "gpu"],
        iterations=2,
        warmup=1,
        frame_size=(32, 32),
        compile_timing=False,
    )
    assert {r["tier"] for r in rows} == {"cpu", "gpu"}
    assert all("mean_ms" in r for r in rows)
    assert set(built) == {"cpu", "gpu"}  # a runner per tier, tier overridden
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_benchmark_pipeline.py -q`
Expected: FAIL — file `tools/benchmark_pipeline.py` does not exist.

- [ ] **Step 3: Implement `tools/benchmark_pipeline.py`**

```python
#!/usr/bin/env python
"""Whole-pipeline benchmark across runtime tiers.

Runs the full InferenceRunner pipeline described by a config file at each of the
three tiers (cpu / gpu / gpu_fast) and reports end-to-end per-frame timing.
Unlike the retired tools/benchmark_models.py, it does NOT benchmark individual
models or arbitrary runtime strings — tiers are the user-facing choice now.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from hydra_suite.core.inference.config import build_inference_config_from_params
from hydra_suite.core.inference.runner import InferenceRunner
from hydra_suite.runtime.resolver import available_tiers, detect_platform, tier_label


# Test seam: a tiny stand-in config so unit tests need no real models.
@dataclasses.dataclass
class _Cfg:
    runtime_tier: str = "cpu"


def _synthetic_frame(frame_size: tuple[int, int]) -> np.ndarray:
    h, w = frame_size
    return np.zeros((h, w, 3), dtype=np.uint8)


def run_pipeline_benchmark(
    config_params: dict,
    *,
    tiers: list[str],
    iterations: int,
    warmup: int,
    frame_size: tuple[int, int],
    compile_timing: bool,
) -> list[dict]:
    """Time the full pipeline per tier. Returns one row dict per tier."""
    rows: list[dict] = []
    frame = _synthetic_frame(frame_size)
    for tier in tiers:
        base = build_inference_config_from_params(config_params)
        cfg = dataclasses.replace(base, runtime_tier=tier)
        row: dict = {"tier": tier}
        t_build0 = time.perf_counter()
        try:
            runner = InferenceRunner(cfg)
        except Exception as e:  # noqa: BLE001 - report, don't crash the whole sweep
            row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            continue
        if compile_timing:
            row["build_s"] = time.perf_counter() - t_build0
        try:
            for _ in range(max(0, warmup)):
                runner.run_realtime(frame)
            samples = []
            for _ in range(max(1, iterations)):
                t0 = time.perf_counter()
                runner.run_realtime(frame)
                samples.append((time.perf_counter() - t0) * 1000.0)
            row["mean_ms"] = float(np.mean(samples))
            row["p50_ms"] = float(np.percentile(samples, 50))
            row["p95_ms"] = float(np.percentile(samples, 95))
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        finally:
            runner.close()
        rows.append(row)
    return rows


def _load_config_params(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to a JSON config file (params dict).")
    parser.add_argument(
        "--tiers",
        nargs="*",
        default=None,
        help="Tiers to benchmark (default: all available on this platform).",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--frame-size", type=int, nargs=2, default=[2000, 2000])
    parser.add_argument(
        "--compile-timing",
        action="store_true",
        help="Also report per-tier runner build time (engine build / export).",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    platform = detect_platform()
    tiers = args.tiers or available_tiers(platform)
    params = _load_config_params(args.config)
    rows = run_pipeline_benchmark(
        params,
        tiers=tiers,
        iterations=args.iterations,
        warmup=args.warmup,
        frame_size=tuple(args.frame_size),
        compile_timing=args.compile_timing,
    )
    for r in rows:
        label = tier_label(r["tier"], platform)
        if "error" in r:
            print(f"{label:22s}  ERROR: {r['error']}")
        else:
            extra = f"  build={r['build_s']:.1f}s" if "build_s" in r else ""
            print(f"{label:22s}  mean={r['mean_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms{extra}")
    if args.output_json:
        args.output_json.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_benchmark_pipeline.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add tools/benchmark_pipeline.py tests/test_benchmark_pipeline.py
git commit -m "feat(tools): tier-based whole-pipeline benchmark CLI"
```

---

### Task 2: Delete the old CLI benchmark and its test

**Files:**
- Delete: `tools/benchmark_models.py`, `tests/test_benchmark_models.py`

- [ ] **Step 1: Confirm nothing imports them**

Run:
```bash
grep -rn "benchmark_models" src/ tools/ tests/ Makefile
```
Expected: matches only inside `tools/benchmark_models.py` / `tests/test_benchmark_models.py` themselves (and possibly a Makefile target — note it for Step 3).

- [ ] **Step 2: Delete**

```bash
git rm tools/benchmark_models.py tests/test_benchmark_models.py
```

- [ ] **Step 3: Remove any Makefile/docs references**

If Step 1 found a `Makefile` target or docs mention, remove/redirect it to `tools/benchmark_pipeline.py`.

- [ ] **Step 4: Verify import health**

Run:
```bash
python -m pytest tests/ -m "not benchmark" -q -k "benchmark"
```
Expected: only `test_benchmark_pipeline.py` collected; PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(tools): delete legacy per-model benchmark CLI + test"
```

---

### Task 3: Remove in-UI benchmarking

**Files:**
- Delete: `src/hydra_suite/trackerkit/benchmarking.py`, `src/hydra_suite/trackerkit/gui/dialogs/benchmark_dialog.py`, `tests/test_trackerkit_benchmarking.py`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py`, `src/hydra_suite/trackerkit/gui/main_window.py`

- [ ] **Step 1: Map every reference**

Run:
```bash
grep -rn "benchmarking\|benchmark_dialog\|BenchmarkDialog\|_open_benchmark_dialog\|collect_active_targets\|run_target_benchmark\|lookup_cached_recommendation" \
  src/hydra_suite/trackerkit
```
Record each hit. Known: `orchestrators/config.py` imports at L34-37 + calls L3665/L3712/L3714; `main_window.py:_open_benchmark_dialog` L1085-1087; `benchmark_dialog.py` L34-42/254/286/317/537.

- [ ] **Step 2: Delete the modules + test**

```bash
git rm src/hydra_suite/trackerkit/benchmarking.py \
       src/hydra_suite/trackerkit/gui/dialogs/benchmark_dialog.py \
       tests/test_trackerkit_benchmarking.py
```

- [ ] **Step 3: Remove the orchestrator wiring**

In `orchestrators/config.py`: delete the benchmarking imports (L34-37) and the methods/calls that use `collect_active_targets` / `lookup_cached_recommendation` / `run_target_benchmark` (around L3665/L3712/L3714) — including any `_open_benchmark_dialog` delegate. Leave `_build_optimizer_detection_cache` (unrelated; owned by Plan 2).

- [ ] **Step 4: Remove the main-window entry point**

In `main_window.py`: delete `_open_benchmark_dialog` (L1085-1087) and the menu action / toolbar button / signal connection that triggers it (grep `_open_benchmark_dialog` and the menu text "Benchmark").

- [ ] **Step 5: Verify the app imports and no dangling refs**

Run:
```bash
grep -rn "benchmarking\|benchmark_dialog\|_open_benchmark_dialog" src/hydra_suite/trackerkit
python -c "import hydra_suite.trackerkit.gui.main_window"
python -m pytest tests/ -m "not benchmark" -q
```
Expected: grep prints nothing (or only unrelated words); no ImportError; suite passes.

- [ ] **Step 6: Commit**

```bash
make format
git add -A
git commit -m "chore(trackerkit): remove in-UI benchmarking (superseded by tier CLI)"
```

---

## Final verification (whole plan)

- [ ] **Step 1: Full suite** — `python -m pytest tests/ -m "not benchmark" -q` → PASS.
- [ ] **Step 2: New tool smoke** — `python tools/benchmark_pipeline.py --help` prints usage.
- [ ] **Step 3: Confirm the old surfaces are gone**

```bash
grep -rn "benchmark_models\|trackerkit/benchmarking\|benchmark_dialog" src/ tools/ tests/
```
Expected: no output.

---

## Self-Review notes

- **Spec coverage (Phase D):** Task 1 builds the tier CLI (Decision 3: config in → whole pipeline per tier → timing + opt-in build timing). Tasks 2–3 delete the old CLI + all in-UI benchmarking + tests.
- **Ordering:** This plan deletes `tests/test_benchmark_models.py`, which hard-requires `YOLOOBBDetector` importable — so this plan must land before Plan 4's `core/detectors` deletion. Noted in Global Constraints.
- **`--compile-benchmark` absorption:** the old flag's engine-build-timing is preserved as `--compile-timing` on the new CLI (per-tier runner build time), matching Decision 3.
- **Type consistency:** `run_pipeline_benchmark(config_params, *, tiers, iterations, warmup, frame_size, compile_timing) -> list[dict]`; rows carry `tier`/`mean_ms`/`p95_ms`/optional `build_s`/`error`. `dataclasses.replace(base_cfg, runtime_tier=tier)` overrides the tier per run.
