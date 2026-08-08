# Identity Overhaul — Phase 3: Evidence as an Inference-Time Artifact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move identity-evidence generation out of the tracking loop and into an `IdentityEvidenceStage` in the inference pass, so calibrated `IdentityEvidence` is a **cached inference-time artifact** written *before* tracking (batch) or inline (realtime). The tracker only *reads* the evidence cache; the tracking-time emitter and the top-1/`split("_")` reconstruction fallback are deleted.

**Architecture:** Every primitive already exists — `IdentityEvidence` (calibrated catalog `log_probs`), `IdentityEvidenceCache` (NPZ sidecar), the structured `(factor_index, class) → catalog_index` mapping + per-factor calibration (in `IdentityEvidenceEmitter`), and raw per-factor softmax in the CNN cache (`CNNCacheHandle`, `raw_probabilities`). This phase relocates the emitter's evidence-building logic to run against the **raw caches** in `InferenceRunner`, resolves the catalog + per-factor calibration **before** the pass, and writes the evidence sidecar as a cache with its own key (catalog + temps), leaving the raw CNN/AprilTag caches untouched. Because the tracker's online decoder already supports reading an evidence sidecar, flipping it to read the inference-written cache — and deleting the emitter — leaves online decisions **byte-identical**. That parity (new-stage evidence == old-emitter evidence) is asserted directly at the task level, then proven end-to-end by the equivalence gate.

**Tech Stack:** Python 3, NumPy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-22-identity-overhaul-consolidated-design.md` — Layer 2 (Evidence layer), Rollout "Phase 3". Builds on Phase 1 (`IdentityCatalog`/`resolve_catalog_spec`/`CatalogEntry.factors`) and Phase 2 (`CNNConfig.calibration_temperature`, `ClassifierMetadata.calibration_temperature`).

## Global Constraints

- **Byte-identical is the gate.** The `IdentityEvidenceStage` must produce evidence identical to what the current `IdentityEvidenceEmitter` produces for the same raw inputs, so online/offline decisions and tracking positions are byte-identical vs the pre-phase branch state. Gate: full equivalence matrix, **MPS + CUDA**, baselined against the **branch commit at the start of Phase 3** (not `legacy/main`, not main-HEAD — the branch has Phases 1-2 on it; isolate *this* phase's delta). Verify CSV row counts > 1.
- **Do not touch raw-cache keys.** The CNN/AprilTag raw caches store raw probabilities and exclude temperature/catalog from their keys (`cnn_cache_key`, `keys.py:230-238`). The evidence sidecar is a **separate** cache with its **own** key (catalog spec signature + per-factor temps + calibration signature). Changing calibration or the catalog must invalidate only the evidence sidecar, never the raw caches.
- **Reuse the emitter's mapping/calibration math; don't reinvent it.** The structured `(factor_index, class)→catalog` mapping and per-factor calibration in `IdentityEvidenceEmitter` (`core/tracking/identity/evidence_emitter.py`) are correct. Extract/share that logic so the inference stage and any residual path cannot diverge. Never `split("_")` a composite label — key off `CatalogEntry.factors`.
- **Isolation.** Work in the existing long-lived worktree `/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/identity-phases-3-7` (branch `feat/identity-phases-3-7`). Do NOT create a new worktree; do NOT merge to main (Phases 3-7 accumulate here; single merge at the end).
- **Dependency direction.** The stage lives in Core (`core/inference/stages/` or `core/individual/identity/`) and imports only Core/Runtime/Utils — never an app layer.
- **Commit as the configured git user.** No `Co-Authored-By: Claude` trailer.
- **Before commit:** `make format` then `make lint` (moderate gate). Revert unrelated isort/black drift. Kill stale `sleap`/`hydra` before heavy runs.
- **Verification:** unit tests on `hydra-mps`; equivalence gate on `hydra-mps` + `hydra-cuda` (mehek).

---

## Current-state anchors (verified by exploration; line numbers may drift — locate by content)

- **Artifact/cache (reuse as-is):** `core/individual/identity/evidence.py` — `IdentityEvidence` (`:37-94`), `from_cnn` (`:141-162`), `from_apriltag` (`:123-139`), `missing` (`:100-121`). `core/individual/identity/cache.py` — `IdentityEvidenceCache` (`save_frame` `:103-141`, `flush` `:143-166`, `load_frame` `:186-231`, path `<base>_identity_evidence_<sig>.npz`).
- **Emitter (source of the logic; deleted at the end):** `core/tracking/identity/evidence_emitter.py` — `_factor_class_to_catalog: dict[(factor_index, class)→list[catalog_idx]]` built `:94-109`; `_factor_log_prob` `:223-273`; `_calibrate_posterior` (per-factor temperature) `:275-290`; `_build_log_probs_from_posteriors` `:292-306`; `build_frame_evidences` `:177-216`; `build_evidence_cache_path` `:398-412`.
- **Raw caches (read by the stage):** `core/inference/cache/store.py` — `CNNCacheHandle.read_frame → list[CNNDetectionPrediction]` (`:332-361`, factors carry `raw_probabilities`); `AprilTagCacheHandle.read_frame → AprilTagResult(tag_ids, det_indices, …)` (`:502-514`).
- **Inference pass:** `core/inference/runner.py` — `InferenceRunner` (`:500`); `run_batch_pass` (`:1014`; pipeline runs `:1066-1072`, caches close in `finally` `:1073-1083`); `run_realtime` (`:614`; raw caches written `:831-851`, `FrameResult` built `:853`). `_open_caches` (`:321-378`), `_CacheSet` (`:97-116`). Runner constructed with only `InferenceConfig`+`cache_dir`+`video_path`+`roi_mask` (`worker.py:1004-1016`).
- **Catalog/calibration resolution (today, in the worker):** `resolve_catalog_spec(p["CNN_CLASSIFIERS"], p["TAG_IDENTITY_LABELS"])` (`worker.py:1842-1847`, from `core/individual/identity/resolve.py:48`); per-factor temps from each CNN cfg (`worker.py` ~`:4259-4272`, `CNNConfig.calibration_temperature`). Both available in `p` **before** `run_batch_pass` (`worker.py:1207`).
- **Tracker consumption (already sidecar-capable):** online feed at `worker.py:2946-3168`; reads sidecar via `_evidence_cache.load_frame` (`:3030-3037`), cache opened by probing `build_evidence_cache_path(_path, label, sig)` for `sig ∈ ("batch","live","")` (`:1583-1610`). AprilTag evidence inline `from_apriltag` (`:3007-3009`); cached-evidence `from_cnn` (`:3054-3062`) — **KEEP**. Top-1/`split("_")` reconstruction fallback (`:3066-3164`) — **DELETE**.
- **Emitter wiring (delete):** construct `worker.py:1471-1483` + builder `_build_cnn_evidence_emitter` `:4192-4315`; fed per-frame `:2460-2472` (via `frame_result_bridge.py:192-205`); flush `:3993-4007`. Dead `self.detected_cnn_cache_paths = {}` `worker.py:153` (+ GUI proxy chain) — delete.
- **Deferred (NOT this phase):** `CNNIdentityCache` class + `augment_trajectories_with_detected_cnn_cache` (still feeds CSV export — Phase 5/retirement); `TrackCNNHistory.majority_class` (already orphaned, no live caller — later cleanup); offline path repointing (Phase 5).

---

## File Structure

**Create:**
- `src/hydra_suite/core/individual/identity/evidence_builder.py` — the shared, Qt-free, tracker-free evidence builder: structured `(factor,class)→catalog` mapping + per-factor calibration + true-softmax → `IdentityEvidence`, over a shared `IdentityCatalog`. (Extracted from the emitter so the inference stage and the emitter cannot diverge.)
- `src/hydra_suite/core/inference/stages/identity_evidence.py` — `IdentityEvidenceStage`: consumes raw CNN/AprilTag cache reads for a frame → `list[IdentityEvidence]`; owns tag→label mapping.
- `src/hydra_suite/core/inference/identity_evidence_key.py` (or add to `cache/keys.py`) — `identity_evidence_cache_key(catalog_spec, per_factor_temps, …)`.
- `tests/identity/test_evidence_builder_parity.py`, `tests/identity/test_identity_evidence_stage.py`, `tests/identity/test_evidence_cache_key.py`.

**Modify:**
- `core/tracking/identity/evidence_emitter.py` — refactor to delegate its mapping/calibration to `evidence_builder.py` (so parity is structural), pending deletion in Task 5.
- `core/inference/runner.py` — accept a resolved catalog + per-factor calibration; invoke `IdentityEvidenceStage` in `run_batch_pass` (after `pipeline.run`, before cache close) and `run_realtime` (after raw caches written).
- `core/inference/config.py` / a small carrier — thread the catalog spec + per-factor temps into the runner (or a dedicated `IdentityEvidenceConfig`).
- `core/tracking/worker.py` — resolve catalog+calibration pre-inference and pass to the runner; **delete** the emitter construct/feed/flush/builder, the top-1/`split("_")` fallback (`:3066-3164`), and the dead `detected_cnn_cache_paths` init; flip online to read the inference-written sidecar.
- GUI proxy chain for `detected_cnn_cache_paths` (`trackerkit/gui/workers/tracking_worker.py:90-91`, `main_window.py:338`, `orchestrators/tracking.py`) — remove the dead proxy.

---

## Interfaces (defined once)

```python
# core/individual/identity/evidence_builder.py
class EvidenceBuilder:
    def __init__(self, catalog: IdentityCatalog, source_name: str,
                 class_labels_per_factor: list[list[str]],
                 calibration: "CalibrationModel | None" = None,
                 calibration_signature: str = "", runtime_signature: str = ""): ...
    # structured (factor_index, class)->catalog map built from `catalog` + factors
    def build_frame_evidences(self, frame_idx: int, det_ids: Sequence[int],
                              per_det_factor_probs: list[list[np.ndarray]]  # [det][factor]=(K_f,) raw softmax
                              ) -> list[IdentityEvidence]: ...

# core/inference/stages/identity_evidence.py
class IdentityEvidenceStage:
    def __init__(self, catalog: IdentityCatalog, cnn_builders: dict[str, EvidenceBuilder],
                 tag_to_label: dict[int, str], tag_source_name: str = "apriltag"): ...
    def evidences_for_frame(self, frame_idx: int, det_ids: Sequence[int],
                            cnn_reads: dict[str, list["CNNDetectionPrediction"]],
                            tag_read: "AprilTagResult | None") -> list[IdentityEvidence]: ...

# key
def identity_evidence_cache_key(catalog_spec: IdentityCatalogSpec,
                                per_factor_temps: Mapping[str, tuple[float, ...]],
                                base_signature: str) -> str: ...  # short content hash
```

---

## Task 1: Evidence-sidecar cache key

A content signature over the catalog spec + per-factor temperatures (+ the raw-cache base signature) so the evidence sidecar invalidates when calibration/catalog change, independently of the raw caches.

**Files:** Create `src/hydra_suite/core/inference/identity_evidence_key.py`; Test `tests/identity/test_evidence_cache_key.py`.

**Interfaces:** Produces `identity_evidence_cache_key(...) -> str`. Consumed by Task 4.

- [ ] **Step 1: Failing test** — same catalog+temps → same key; changed temp → different key; changed catalog label → different key; deterministic across runs; short hex string.

```python
# tests/identity/test_evidence_cache_key.py
from hydra_suite.core.individual.identity.spec import CatalogEntry, IdentityCatalogSpec
from hydra_suite.core.inference.identity_evidence_key import identity_evidence_cache_key


def _spec(label="red_big"):
    return IdentityCatalogSpec(entries=(
        CatalogEntry(display_label=label, factors=(("color", "red"), ("size", "big")), source="cnn"),
    ))


def test_stable_and_sensitive():
    a = identity_evidence_cache_key(_spec(), {"cnn0": (1.5,)}, "vidsig")
    b = identity_evidence_cache_key(_spec(), {"cnn0": (1.5,)}, "vidsig")
    assert a == b and isinstance(a, str) and len(a) >= 8
    assert identity_evidence_cache_key(_spec(), {"cnn0": (2.0,)}, "vidsig") != a
    assert identity_evidence_cache_key(_spec("blue_big"), {"cnn0": (1.5,)}, "vidsig") != a
    assert identity_evidence_cache_key(_spec(), {"cnn0": (1.5,)}, "other") != a
```

- [ ] **Step 2: Run — FAIL** (module missing).

- [ ] **Step 3: Implement**

```python
# src/hydra_suite/core/inference/identity_evidence_key.py
"""Content signature for the identity-evidence sidecar cache.

Independent of the raw CNN/AprilTag cache keys: it folds in the catalog domain
and the per-factor calibration temperatures so a recalibration or catalog change
invalidates only the evidence sidecar, never the raw caches.
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

from hydra_suite.core.individual.identity.spec import IdentityCatalogSpec


def identity_evidence_cache_key(
    catalog_spec: IdentityCatalogSpec,
    per_factor_temps: Mapping[str, tuple[float, ...]],
    base_signature: str,
) -> str:
    payload = {
        "catalog": catalog_spec.to_dict(),
        "temps": {k: [round(float(t), 6) for t in v] for k, v in sorted(per_factor_temps.items())},
        "base": str(base_signature),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]
```

- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: `make format && make lint`; commit** `feat(identity): identity-evidence sidecar cache key (catalog + per-factor temps)`.

---

## Task 2: `EvidenceBuilder` (shared) + parity with the emitter

Extract the emitter's structured-mapping + calibration + evidence-construction into a Qt-free, tracker-free `EvidenceBuilder` over a shared `IdentityCatalog`, and refactor `IdentityEvidenceEmitter` to delegate to it. A **parity test** asserts the builder reproduces the emitter's evidence for identical inputs — this is the linchpin that makes the later deletion byte-identical.

**Files:** Create `core/individual/identity/evidence_builder.py`; Modify `core/tracking/identity/evidence_emitter.py` (delegate); Test `tests/identity/test_evidence_builder_parity.py`.

**Interfaces:** Produces `EvidenceBuilder` (Interfaces block). Consumed by Task 3's stage + the emitter.

- [ ] **Step 1: Study the emitter, write the parity test.** READ `evidence_emitter.py` fully. The test builds an `EvidenceBuilder` and an `IdentityEvidenceEmitter` with the SAME `class_labels_per_factor` + `calibration` + catalog, feeds both the SAME synthetic per-factor raw-softmax for a few detections, and asserts the resulting `IdentityEvidence.log_probs` are equal (`np.allclose` with atol=0, or exact) and `source_name`/`observed_mask` match. Cover: single-factor, multi-factor composite (incl. a class name containing `_` so the structured path is exercised and `split("_")` would fail), and a per-factor temperature ≠ 1.

```python
# tests/identity/test_evidence_builder_parity.py — skeleton (fill exact emitter ctor args from the file)
import numpy as np
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.evidence_builder import EvidenceBuilder
from hydra_suite.core.tracking.identity.evidence_emitter import IdentityEvidenceEmitter


def test_builder_matches_emitter_multifactor_with_underscore(tmp_path):
    labels = [["dark_red", "blue"], ["big", "small"]]      # class with "_" -> split() would corrupt
    catalog = IdentityCatalog.from_labels(["dark_red_big", "dark_red_small", "blue_big", "blue_small"])
    # per-det per-factor raw softmax
    probs = [[np.array([0.7, 0.3]), np.array([0.6, 0.4])],
             [np.array([0.2, 0.8]), np.array([0.9, 0.1])]]
    builder = EvidenceBuilder(catalog, "cnn0", labels, calibration=None)
    ev_b = builder.build_frame_evidences(5, [10, 11], probs)
    # emitter path (posteriors == calibrated probs; no calibration): reuse its build_frame_evidences
    emitter = IdentityEvidenceEmitter(cache_path=str(tmp_path / "e.npz"), source_name="cnn0",
                                      class_labels_per_factor=labels, calibration=None)
    ev_e = emitter.build_frame_evidences(5, _predictions_from(probs, labels), posteriors=probs, detection_ids=[10, 11])
    for b, e in zip(ev_b, ev_e):
        assert np.array_equal(b.log_probs, e.log_probs)
```

(You will need `_predictions_from(...)` to build the `ClassPrediction` shape the emitter expects — READ `build_frame_evidences`/`build_frame_evidences`'s `predictions` param and mirror it. If constructing the emitter's exact `predictions` shape is heavy, assert parity against the emitter's lower-level `_build_log_probs_from_posteriors` instead, and say so.)

- [ ] **Step 2: Run — FAIL** (module missing).

- [ ] **Step 3: Implement `EvidenceBuilder`** by lifting the emitter's `_factor_class_to_catalog` construction (`:94-109`), `_factor_log_prob` (`:223-273`), `_calibrate_posterior` (`:275-290`), `_build_log_probs_from_posteriors` (`:292-306`), and `build_frame_evidences` (`:177-216`) into the new class, parameterized by a passed-in `IdentityCatalog` (instead of building its own internal catalog). Keep the math **identical**. Then **refactor `IdentityEvidenceEmitter` to delegate** to `EvidenceBuilder` (construct one internally from its cartesian catalog + calibration; forward `build_frame_evidences`). The emitter keeps its cache-writing responsibility until Task 5 deletes it.

- [ ] **Step 4: Run — PASS** (parity holds across all cases). Run full `tests/identity/` — no regression.

- [ ] **Step 5: `make format && make lint`; commit** `refactor(identity): shared EvidenceBuilder; emitter delegates (parity-tested)`.

---

## Task 3: `IdentityEvidenceStage` over raw cache reads

The inference-time producer: given per-frame raw CNN predictions + AprilTag result + the shared catalog + per-factor `EvidenceBuilder`s + tag→label, return `list[IdentityEvidence]`. Owns the single tag→catalog mapping.

**Files:** Create `core/inference/stages/identity_evidence.py`; Test `tests/identity/test_identity_evidence_stage.py`.

**Interfaces:** Consumes Task 2 (`EvidenceBuilder`), `IdentityCatalog.apriltag_log_prior`. Produces `IdentityEvidenceStage`. Consumed by Task 4.

- [ ] **Step 1: Failing test** — construct a stage with a 1-CNN `EvidenceBuilder` + a `tag_to_label`, feed synthetic `cnn_reads` (`list[CNNDetectionPrediction]` with `raw_probabilities`) + an `AprilTagResult`; assert: CNN dets get `from_cnn` evidence with the builder's `log_probs`; tagged dets get `from_apriltag` evidence equal to `catalog.apriltag_log_prior(tag_id, tag_to_label)`; det ordering/ids preserved; a det with no CNN factor + no tag is absent (or `missing`, matching the emitter's contract — READ the emitter to decide which). Build the `CNNDetectionPrediction`/`CNNFactorPrediction` + `AprilTagResult` shapes per `core/inference/result.py` / `stages/apriltag.py`.

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement.** For each CNN phase read, pull per-det per-factor `raw_probabilities` (aligned to det_ids) and call the phase's `EvidenceBuilder.build_frame_evidences`. For the AprilTag read, build `tag_to_label` once (from `TAG_IDENTITY_LABELS`, the `enumerate` index→label pattern at `worker.py:2706`/`:2985`) and emit `IdentityEvidence.from_apriltag(frame_idx, det_id, catalog.apriltag_log_prior(tag_id, tag_to_label))` per tagged det. Merge per-frame CNN + tag evidences. Do NOT `split("_")` anywhere.

- [ ] **Step 4: Run — PASS.** Full `tests/identity/` — no regression.

- [ ] **Step 5: `make format && make lint`; commit** `feat(inference): IdentityEvidenceStage builds evidence from raw caches`.

---

## Task 4: Wire the stage into `InferenceRunner` (batch + realtime) + thread catalog/calibration

Resolve the catalog + per-factor calibration before the pass (in the worker) and thread them to the runner; write the evidence sidecar during inference.

**Files:** Modify `core/inference/runner.py`, a config carrier (e.g. `core/inference/config.py` or a new `IdentityEvidenceConfig`), and `core/tracking/worker.py` (resolve + pass; do NOT delete the emitter yet). Test: a runner-level integration test on a tiny synthetic cache dir, or an assertion that the written sidecar's evidence equals the emitter's for the same clip (see Gate).

**Interfaces:** Consumes Task 1 (key), Task 3 (stage). Produces the inference-written sidecar. Consumed by Task 5.

- [ ] **Step 1:** Add an optional `identity_evidence` config to the runner carrying: the resolved `IdentityCatalogSpec`, per-phase `class_names_per_factor`, per-phase `CalibrationModel` (per-factor temps), and `tag_to_label`/`TAG_IDENTITY_LABELS`. When absent (no identity configured), the stage is a no-op (no behavior change).

- [ ] **Step 2 (batch):** In `run_batch_pass`, after `pipeline.run(...)` completes and before closing caches (the `finally`), if identity-evidence config is present: read back each frame's raw CNN/AprilTag caches (`caches.cnn[i].read_frame`, `caches.apriltag.read_frame`), run `IdentityEvidenceStage.evidences_for_frame`, and write them to an `IdentityEvidenceCache` at `build_evidence_cache_path(<cache base>, "batch", identity_evidence_cache_key(...))`. Flush before caches close.

- [ ] **Step 3 (realtime):** In `run_realtime`, after the raw caches for the frame are written (`:851`) and before `FrameResult` build (`:853`), run the stage for that frame and `save_frame` into a realtime evidence cache (flushed at pass end). Identical evidence contract to batch.

- [ ] **Step 4 (worker threading):** In `worker.py`, before `run_batch_pass`/`run_realtime`, resolve `catalog_spec = resolve_catalog_spec(p["CNN_CLASSIFIERS"], p["TAG_IDENTITY_LABELS"])`, build per-phase `CalibrationModel`s (reuse the existing `worker.py:4259-4272` resolution), and pass the identity-evidence config into the runner. The emitter still runs (deleted in Task 5) — both writing evidence is fine transiently; the tracker still reads the emitter's "live" cache until Task 5 flips it.

- [ ] **Step 5:** Test at the runner level (synthetic cache dir with a couple of frames of raw CNN probs + tags → assert the sidecar exists and `load_frame` returns evidence equal to a direct `IdentityEvidenceStage` call). Run `tests/identity/` + any inference-runner tests — no regression.

- [ ] **Step 6: `make format && make lint`; commit** `feat(inference): write identity-evidence sidecar during the inference pass (batch + realtime)`.

---

## Task 5: Flip the tracker to the inference sidecar; delete the emitter + reconstruction fallback

The honesty-relevant cutover: the tracker reads the inference-written sidecar; the tracking-time emitter and the top-1/`split("_")` fallback are removed.

**Files:** Modify `core/tracking/worker.py` (+ `frame_result_bridge.py`, GUI proxy chain). Test: the equivalence gate is the primary guard; add a focused assertion that with the inference sidecar present, the online decoder's per-frame evidence equals what the emitter produced (can reuse Task 2/4 parity harness on one fixture).

- [ ] **Step 1:** Make the worker's evidence-cache open (`:1583-1610`) find the **inference-written** sidecar (the `"batch"`/realtime-signature file from Task 4). Confirm the online feed (`:3030-3037`) reads it.

- [ ] **Step 2:** Delete the emitter wiring: construct `:1471-1483`, `_build_cnn_evidence_emitter` `:4192-4315`, per-frame feed `:2460-2472` (+ the `evidence_emitter=` param/branch in `frame_result_bridge.py:192-205`), flush `:3993-4007`. Delete the whole top-1/`split("_")` reconstruction fallback `:3066-3164` (keep the AprilTag `:3007` + cached-evidence `:3054` constructions). Delete the dead `self.detected_cnn_cache_paths = {}` (`:153`) and its GUI proxy chain (`tracking_worker.py:90-91`, `main_window.py:338`, `orchestrators/tracking.py` references).

- [ ] **Step 3: Grep-guard** — `grep -n "IdentityEvidenceEmitter\|_evidence_emitters\|_build_cnn_evidence_emitter\|detected_cnn_cache_paths\|split(\"_\")\|_per_factor_dist" src/hydra_suite/core/tracking/worker.py` → no surviving references (except unrelated). If `evidence_emitter.py` now has no importers, delete the file too.

- [ ] **Step 4:** Run `tests/identity/` + tracking-adjacent tests. Then the equivalence smoke (fastest clips) — see Gate.

- [ ] **Step 5: `make format && make lint`; commit** `refactor(identity): tracker reads inference evidence sidecar; delete tracking-time emitter + top-1/split reconstruction`.

---

## Phase-End Gate

- [ ] **Full equivalence — MPS.** Baseline = the branch commit at the START of Phase 3 (record it before Task 1); current = branch HEAD. `MAIN_SRC=<worktree copy at phase-start SHA>/src WT_SRC=<worktree>/src` (make a detached worktree at the phase-start SHA for the baseline src). Every clip EQUIVALENT at its determinism floor for `_forward.csv` + `_tracking_final.csv`; identical row counts; 0 unmatched. This proves the evidence relocation is byte-identical. **Verify CSV row counts > 1.**
- [ ] **Full equivalence — CUDA (mehek)** per the CLAUDE.md recipe against the phase-start baseline (push the branch; worktrees at phase-start SHA + branch HEAD; `setsid` detached run; re-fetch fixtures with `PYTHONPATH=$PWD/src`).
- [ ] **Suite delta gate:** `python -m pytest tests/identity/ -v` green.
- [ ] **Evidence-provenance check:** confirm the evidence sidecar is written by the inference pass BEFORE tracking (e.g. it exists after a `cache-only`/inference-only run, before any tracking) — the core Phase-3 property.

---

## Self-Review (against the spec — Layer 2 + Phase 3)

- "`InferenceRunner` gains an `IdentityEvidenceStage` … produces `IdentityEvidence` … writes the `IdentityEvidenceCache` sidecar" → Tasks 3, 4. ✅
- "Non-realtime: written in `run_batch_pass` … before tracking. Realtime: emitted inline in `run_realtime`, identical contract" → Task 4 Steps 2/3. ✅
- "owns the single factor→catalog mapping … true per-factor softmax (no top-1 reconstruction) … structured factor keys" → Tasks 2/3 (structured `(factor,class)` map; parity test with a `_`-containing class). ✅
- "Deleted: tracking-time `IdentityEvidenceEmitter` … the top-1 pseudo-distribution reconstruction … `detected_cnn_cache_paths`" → Task 5. ✅
- **Deferred (documented):** orphaned V3 `CNNIdentityCache` (still feeds CSV export → retire in Phase 5/7), `TrackCNNHistory` majority-vote (already orphaned, later cleanup), offline path repointing (Phase 5). ✅ — these are explicitly NOT in Phase 3 to keep the byte-identical gate clean.
- **Placeholder scan:** novel pieces (key, builder) carry real code; wiring/deletion tasks carry exact anchors + a `READ <file> first` where the real shape must be matched (emitter `predictions` shape, cache read shapes). Parity + grep-guard + equivalence gate are the safety nets.
- **Type consistency:** `EvidenceBuilder.build_frame_evidences(det_ids, per_det_factor_probs)` and `IdentityEvidenceStage.evidences_for_frame(...)` return `list[IdentityEvidence]`; the sidecar key signature is used identically in Tasks 1 and 4.

**Risk to watch at execution:** the emitter today may, in some path, feed evidence from top-1 predictions rather than true posteriors — if so, switching to true per-factor softmax would change evidence and break byte-identical. Task 2's parity test MUST use the same input contract the live pipeline actually feeds the emitter; if the live pipeline feeds top-1 (not posteriors), that is itself the bug Phase 3 fixes and the "byte-identical" expectation must be renegotiated with the user (surface it, don't silently accept a diff). Confirm during Task 2 which the live pipeline uses (posteriors vs predictions) and report before proceeding to deletion.
