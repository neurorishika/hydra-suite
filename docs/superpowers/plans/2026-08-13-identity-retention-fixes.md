# Identity Retention Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all 13 incremental fixes from the identity audit, each gated by an objective test that identity retention actually improves.

**Architecture:** A synthetic retention-scenario harness (driving the real `OnlineIdentityDecoder` / `TrackAssigner` with known ground truth) encodes each fix's target behavior as a `strict=True` xfail test; each fix task flips its xfail to a pass. A fixture-level metrics CLI measures flips/teleports/unknown-fraction on the equivalence fixtures for the cumulative before/after gate. All new behavior is parameterized so legacy behavior stays reachable (ablation) and non-identity clips stay byte-identical.

**Tech Stack:** Python 3.11, numpy, pandas, pytest, existing `tools/equivalence/` harness, conda env `hydra-mps` (this box) + `hydra-cuda` (mehek).

**Spec:** `docs/superpowers/specs/2026-08-13-identity-audit.md` (findings F1–F17, fixes §6 items 1–13). Read §4 and §6 before starting.

## Global Constraints

- **Worktree isolation:** all work happens in a worktree branched from local HEAD: `git worktree add .worktrees/identity-fixes -b feat/identity-retention-fixes HEAD`. All commands below run from `.worktrees/identity-fixes` unless stated.
- **Test command shape:** `PYTHONPATH=$PWD/src conda run -n hydra-mps python -m pytest <file>::<test> -v` (worktree tests need `PYTHONPATH=<wt>/src`; never run the whole `tests/` tree — the classkit modal-dialog hang makes it never finish; run per-file).
- **Commits:** commit as the configured git user; do NOT add a `Co-Authored-By: Claude` trailer.
- **No Artifacts.** Results are files in the repo.
- **Every new knob has a legacy value** that reproduces pre-fix behavior: `IDENTITY_PER_FRAME_EVIDENCE_CAP=0` (off), `IDENTITY_PROB_FLOOR=0` (off), `IDENTITY_EVIDENCE_TAU=1` (off), `IDENTITY_COMMIT_REVISION_MIN_FRAMES=1`, `IDENTITY_REJOIN_CONFIRM_FRAMES=1`, `IDENTITY_SPLIT_ON_REALTIME_SWITCH=False`. `MAX_VELOCITY_ZSCORE` default stays `0.0` (two-sided |z| is implemented but not default-enabled).
- **Non-identity regression gate:** `fly_obb` and `worm_bgsub` equivalence clips must stay byte-identical vs the branch point (they never execute identity code). Identity clips (`emi_obb_identity`, `ant_cnn_identity`) are EXPECTED to differ — their gate is the metrics CLI (Task 2), not byte-equality.
- **Before any heavy run** (Task 16): kill dead/stale sleap/hydra processes only; never touch other processes. conda env must be active or CSVs come out empty and falsely pass — always verify `wc -l` > 1 on produced CSVs.
- The retention test suite (`tests/identity/test_retention_benchmark.py`) must be fully green (no remaining strict xfails that a completed task was supposed to flip, no regressions in the invariant tests) at the end of every task's commit.

## Fix → Task map (audit §6 item → task)

| Audit fix | Finding | Task |
|---|---|---|
| harness (new) | — | 1, 2 |
| 1 slot-lock sign | F3 | 3 |
| 12 emit hardcoded knobs | — | 4 |
| 2 wire cap/floor | F1, F2 | 5 |
| 5 commit-revision debounce | F4, F8 | 6 |
| 3 evidence tempering | F1 | 7 |
| 4 unknown emission | F2 | 8 |
| 6 rejoin hardening | F7 | 9 |
| 13 respawn prior space-keyed + TTL | F9 | 10 |
| 7 evidence double-use / twin likelihoods | F6 | 11 |
| 8 solver online prior source | F12 | 12 |
| 9 raw-evidence support (tempered sum) | F13 | 13 |
| 10 split at realtime switches | F14 | 14 |
| 11 gap-scaled veto + two-sided z | F15 | 15 |
| cumulative pipeline gate | all | 16 |

---

### Task 1: Retention scenario harness

**Files:**
- Create: `tests/identity/retention_scenarios.py`
- Create: `tests/identity/test_retention_benchmark.py`

**Interfaces:**
- Produces: `make_catalog()`, `make_decoder(params)`, `cnn_evidence(catalog, frame, top_label, conf)`, `drive_constant(decoder, slot, label, conf, frames, start=0) -> list[dict]`, `count_committed_flips(history) -> int`, `scenario_isolated_flip(params=None, wrong_frames=12, wrong_conf=0.6, tail_frames=40) -> dict`, `scenario_ambiguity(params=None) -> dict`, `scenario_cold_commit(params=None) -> int|None`, `scenario_true_swap(params=None) -> int|None`, `run_rejoin(det_xy, det_top_conf, lost_frames, params_extra=None, streak_calls=1) -> list`. Later tasks consume these exact names.
- Consumes: `OnlineIdentityDecoder`, `IdentityCatalog`, `IdentityEvidence`, `TrackAssigner` (production code, unmodified).

- [ ] **Step 1: Write the scenario library**

```python
# tests/identity/retention_scenarios.py
"""Synthetic identity-retention scenarios driving the REAL online decoder and
assigner with known ground truth. Used by test_retention_benchmark.py: each
audit fix flips one strict-xfail target test to passing."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.evidence import IdentityEvidence
from hydra_suite.core.individual.identity.online import OnlineIdentityDecoder

LABELS = ["A", "B", "C", "D"]


def make_catalog() -> IdentityCatalog:
    return IdentityCatalog.from_labels(LABELS)


def make_decoder(params: Optional[dict[str, Any]] = None) -> OnlineIdentityDecoder:
    return OnlineIdentityDecoder(make_catalog(), dict(params or {}))


def cnn_evidence(catalog, frame, top_label, conf, source="cnn_bench"):
    """Evidence exactly as catalog.cnn_log_prior builds it in production."""
    known = list(catalog.labels[1:])
    n_other = max(len(known) - 1, 1)
    probs = np.array(
        [conf if lbl == top_label else (1.0 - conf) / n_other for lbl in known]
    )
    lp = catalog.cnn_log_prior(probs, known)
    return IdentityEvidence.from_cnn(frame, 0, source, lp)


def drive_constant(decoder, slot, label, conf, frames, start=0):
    """Feed `frames` frames of constant evidence to one slot; return history."""
    cat = decoder._catalog
    out = []
    for t in range(start, start + frames):
        evs = {slot: [cnn_evidence(cat, t, label, conf)]} if label else {slot: []}
        a = decoder.update_frame(t, [slot], evs)[0]
        bel = decoder.get_belief(slot)
        probs = decoder._posterior_probs(bel)
        out.append(
            {
                "frame": t,
                "label": a.label,
                "committed": a.committed,
                "committed_label": bel.committed_label,
                "p_unknown": float(probs[0]),
                "p": {l: float(probs[cat.index_of(l)]) for l in cat.labels[1:]},
            }
        )
    return out


def count_committed_flips(history):
    """Transitions of committed_label between two different non-None labels."""
    flips = 0
    prev = None
    for row in history:
        cur = row["committed_label"]
        if prev is not None and cur is not None and cur != prev:
            flips += 1
        if cur is not None:
            prev = cur
    return flips


def scenario_isolated_flip(params=None, wrong_frames=12, wrong_conf=0.6, tail_frames=40):
    """Committed isolated track gets a wrong-evidence streak, then truth resumes."""
    dec = make_decoder(params)
    h1 = drive_constant(dec, 0, "A", 0.9, 60)
    h2 = drive_constant(dec, 0, "B", wrong_conf, wrong_frames, start=60)
    h3 = drive_constant(dec, 0, "A", 0.9, tail_frames, start=60 + wrong_frames)
    hist = h1 + h2 + h3
    return {
        "flips": count_committed_flips(hist),
        "final_committed": hist[-1]["committed_label"],
        "flipped_during_streak": any(r["committed_label"] == "B" for r in h2 + h3),
    }


def scenario_ambiguity(params=None, frames=30, conf=0.30):
    """Near-uniform evidence: the system should stay uncertain, not commit."""
    dec = make_decoder(params)
    hist = drive_constant(dec, 0, "A", conf, frames)
    return {
        "p_unknown_end": hist[-1]["p_unknown"],
        "committed": hist[-1]["committed"],
        "max_known_p_end": max(hist[-1]["p"].values()),
    }


def scenario_cold_commit(params=None, conf=0.9, max_frames=60):
    """Frames until commitment under sustained strong genuine evidence."""
    dec = make_decoder(params)
    hist = drive_constant(dec, 0, "A", conf, max_frames)
    for i, row in enumerate(hist):
        if row["committed"]:
            return i + 1
    return None


def scenario_true_swap(params=None, settle=60, crossed=40):
    """Two committed slots receive each other's evidence; frames to correction."""
    dec = make_decoder(params)
    cat = dec._catalog
    for t in range(settle):
        dec.update_frame(
            t,
            [0, 1],
            {0: [cnn_evidence(cat, t, "A", 0.9)], 1: [cnn_evidence(cat, t, "B", 0.9)]},
        )
    for i in range(crossed):
        t = settle + i
        dec.update_frame(
            t,
            [0, 1],
            {0: [cnn_evidence(cat, t, "B", 0.9)], 1: [cnn_evidence(cat, t, "A", 0.9)]},
        )
        b0, b1 = dec.get_belief(0), dec.get_belief(1)
        if b0.committed_label == "B" and b1.committed_label == "A":
            return i + 1
    return None


class _FakeKF:
    def __init__(self, positions):
        self.X = np.asarray(positions, dtype=np.float64)


def run_rejoin(det_xy, det_top_conf, lost_frames, params_extra=None, streak_calls=1):
    """Drive TrackAssigner._assign_respawn's identity-rejoin branch directly.

    One committed-lost slot 0 (belief ~0.96 on 'A', last seen at origin) and
    one unassigned detection at det_xy whose evidence supports 'A' with
    det_top_conf. Returns identity_rejoin_pairs from the LAST call.
    `streak_calls` repeats the call to exercise confirmation windows."""
    from hydra_suite.core.assigners.hungarian import TrackAssigner

    cat = make_catalog()
    params = {
        "MAX_DISTANCE_THRESHOLD": 100.0,
        "REFERENCE_BODY_SIZE": 20.0,
        "RESIZE_FACTOR": 1.0,
        "KALMAN_MAX_VELOCITY_MULTIPLIER": 2.0,
        **(params_extra or {}),
    }
    assigner = TrackAssigner(params)
    log_post = np.log(np.array([0.01, 0.96, 0.01, 0.01, 0.01]))
    known = list(cat.labels[1:])
    n_other = len(known) - 1
    probs = np.array(
        [det_top_conf if l == "A" else (1 - det_top_conf) / n_other for l in known]
    )
    det_ll = cat.cnn_log_prior(probs, known)
    pairs = []
    for _ in range(max(1, int(streak_calls))):
        result = assigner._assign_respawn(
            cost=np.full((1, 1), 1e9),
            N=1,
            meas=[np.array([det_xy[0], det_xy[1], 0.0])],
            track_states=["lost"],
            tracking_continuity=[0],
            kf_manager=_FakeKF([[0.0, 0.0, 0.0, 0.0, 0.0]]),
            association_data={
                "identity_detection_log_likelihoods": [det_ll],
                "identity_track_log_posteriors": {0: log_post},
            },
            committed_slot_identities={0: "A"},
            missed_frames=[int(lost_frames)],
        )
        pairs = result[2]
    return pairs
```

- [ ] **Step 2: Write the benchmark test file**

Invariant tests (must pass NOW and stay passing after every fix — they guard against over-damping) plus strict-xfail target tests (encode post-fix behavior; each fix task removes its xfail marker):

```python
# tests/identity/test_retention_benchmark.py
"""Identity-retention benchmark: invariants + per-fix targets.

Target tests are strict xfails encoding the audit's desired behavior
(docs/superpowers/specs/2026-08-13-identity-audit.md §6). Each fix task in
docs/superpowers/plans/2026-08-13-identity-retention-fixes.md removes exactly
its own xfail marker. Invariant tests must never break."""
import numpy as np
import pytest

from tests.identity.retention_scenarios import (
    drive_constant,
    make_decoder,
    run_rejoin,
    scenario_ambiguity,
    scenario_cold_commit,
    scenario_isolated_flip,
    scenario_true_swap,
)

# ----------------------------- invariants -----------------------------


def test_invariant_cold_commit_is_bounded():
    frames = scenario_cold_commit(conf=0.9, max_frames=60)
    assert frames is not None and frames <= 60


def test_invariant_true_swap_is_corrected():
    assert scenario_true_swap() is not None


def test_invariant_sustained_wrong_evidence_eventually_corrects():
    m = scenario_isolated_flip(wrong_frames=150, wrong_conf=0.9, tail_frames=0)
    assert m["flipped_during_streak"]


def test_invariant_unique_committed_labels():
    dec = make_decoder()
    from tests.identity.retention_scenarios import cnn_evidence

    cat = dec._catalog
    for t in range(60):
        dec.update_frame(
            t,
            [0, 1],
            {0: [cnn_evidence(cat, t, "A", 0.9)], 1: [cnn_evidence(cat, t, "A", 0.9)]},
        )
    labels = [dec.get_belief(s).committed_label for s in (0, 1)]
    committed = [l for l in labels if l]
    assert len(committed) == len(set(committed))


def test_invariant_near_supported_claim_rejoins():
    pairs = run_rejoin(det_xy=(60.0, 0.0), det_top_conf=0.9, lost_frames=10,
                       streak_calls=3)
    assert pairs == [(0, 0)]


# ------------------------- per-fix targets ---------------------------


@pytest.mark.xfail(strict=True, reason="F3 slot-lock sign bug — fixed by Task 3")
def test_target_slot_lock_bias_boosts_locked_label():
    dec = make_decoder()
    drive_constant(dec, 0, "A", 0.9, 40)
    bel = dec.get_belief(0)
    assert bel.slot_lock_label == "A"
    ia = dec._catalog.index_of("A")
    raw = float(dec._posterior_probs(bel)[ia])
    if hasattr(dec, "_assignment_probs"):
        biased = float(dec._assignment_probs(bel)[ia])
    else:  # pre-fix path: persistent in-place bias
        dec._apply_slot_lock_bias(bel)
        biased = float(dec._posterior_probs(bel)[ia])
    assert biased >= raw


@pytest.mark.xfail(strict=True, reason="F4 vacuous revision gate — fixed by Task 6")
def test_target_short_strong_wrong_burst_does_not_revise():
    m = scenario_isolated_flip(wrong_frames=5, wrong_conf=0.9)
    assert m["flips"] == 0
    assert m["final_committed"] == "A"


@pytest.mark.xfail(strict=True, reason="F1 uncapped correlated fusion — fixed by Tasks 5+7")
def test_target_moderate_wrong_streak_does_not_flip():
    m = scenario_isolated_flip(wrong_frames=12, wrong_conf=0.6)
    assert m["flips"] == 0
    assert m["final_committed"] == "A"


@pytest.mark.xfail(strict=True, reason="F2 unknown annihilation — fixed by Task 8")
def test_target_ambiguous_evidence_keeps_unknown_reachable():
    m = scenario_ambiguity(frames=30, conf=0.30)
    assert m["p_unknown_end"] >= 0.05
    assert not m["committed"]


@pytest.mark.xfail(strict=True, reason="F7 single-frame teleport rejoin — fixed by Task 9")
def test_target_far_single_frame_claim_does_not_rejoin():
    # 20 body-lengths away, 10 lost frames, one 0.55-confidence frame:
    # linear budget (600px) admits it today; sqrt-budget (~190px) must not.
    pairs = run_rejoin(det_xy=(400.0, 0.0), det_top_conf=0.55, lost_frames=10,
                       streak_calls=1)
    assert pairs == []


@pytest.mark.xfail(strict=True, reason="F9 committed-lost never expires — fixed by Task 10")
def test_target_committed_lost_slot_expires():
    dec = make_decoder({"IDENTITY_RESPAWN_PRIOR_MAX_GAP": 30})
    drive_constant(dec, 0, "A", 0.9, 40)
    for _ in range(31):
        dec.decay_absent_slot_beliefs([0])
    assert not dec.get_belief(0).committed
```

- [ ] **Step 3: Run the suite — invariants pass, targets xfail**

Run: `PYTHONPATH=$PWD/src conda run -n hydra-mps python -m pytest tests/identity/test_retention_benchmark.py -v`
Expected: 5 passed, 6 xfailed, 0 failed. If any invariant fails or any target unexpectedly passes (XPASS = strict-xfail failure), STOP: the audit's baseline claim is wrong for that scenario — re-derive the scenario numbers against the actual behavior before proceeding (adjust streak lengths/confidences, not the assertion's direction).

- [ ] **Step 4: Commit**

```bash
git add tests/identity/retention_scenarios.py tests/identity/test_retention_benchmark.py
git commit -m "test(identity): retention benchmark harness with per-fix xfail targets"
```

---

### Task 2: Fixture identity-metrics CLI

**Files:**
- Create: `tools/identity_bench/__init__.py` (empty)
- Create: `tools/identity_bench/identity_metrics.py`
- Test: `tests/identity/test_identity_metrics.py`

**Interfaces:**
- Produces: `compute_identity_metrics(df: pd.DataFrame, body_size: float) -> dict` and CLI `python tools/identity_bench/identity_metrics.py <csv> [--body-size 20]` printing a JSON dict with keys `n_rows`, `n_tracks`, `realtime_flips_per_1k`, `final_label_switches`, `teleports_per_1k`, `unknown_fraction`. Task 16 consumes this CLI.

- [ ] **Step 1: Write the failing test**

```python
# tests/identity/test_identity_metrics.py
import pandas as pd

from tools.identity_bench.identity_metrics import compute_identity_metrics


def _df(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "TrajectoryID", "FrameID", "X", "Y",
            "IdentityRealtimeLabel", "IdentityFinalLabel",
        ],
    )


def test_flip_and_teleport_counting():
    rows = []
    # track 1: A for 5 frames, then B for 5 frames (1 realtime flip),
    # with a 200px jump between frames 4 and 5 (1 teleport at body=20 -> 5x=100)
    for f in range(10):
        x = 0.0 if f < 5 else 200.0
        lbl = "A" if f < 5 else "B"
        rows.append((1, f, x, 0.0, lbl, "A"))
    m = compute_identity_metrics(_df(rows), body_size=20.0)
    assert m["n_rows"] == 10 and m["n_tracks"] == 1
    assert m["realtime_flips_per_1k"] == 100.0  # 1 flip / 10 rows * 1000
    assert m["teleports_per_1k"] == 100.0
    assert m["final_label_switches"] == 0
    assert m["unknown_fraction"] == 0.0


def test_unknown_fraction_counts_blank_and_unknown():
    rows = [(1, f, 0.0, 0.0, lbl, "") for f, lbl in enumerate(["A", "", "unknown", "A"])]
    m = compute_identity_metrics(_df(rows), body_size=20.0)
    assert m["unknown_fraction"] == 0.5
```

- [ ] **Step 2: Run to verify it fails** — `PYTHONPATH=$PWD/src:$PWD conda run -n hydra-mps python -m pytest tests/identity/test_identity_metrics.py -v` — expected: ImportError/ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# tools/identity_bench/identity_metrics.py
"""Identity-retention metrics over a tracking CSV (forward or final).

Counts within-trajectory realtime-label flips, final-label switches,
displacement spikes (> 5x body size per frame step), and the unknown-label
fraction. Used as the before/after pipeline gate for the identity fixes
(docs/superpowers/plans/2026-08-13-identity-retention-fixes.md Task 16)."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

_UNKNOWN = {"", "unknown", "nan", "none"}


def _label_switches(series: pd.Series) -> int:
    vals = [str(v).strip() for v in series.tolist()]
    known = [v for v in vals if v.lower() not in _UNKNOWN]
    return sum(1 for a, b in zip(known, known[1:]) if a != b)


def compute_identity_metrics(df: pd.DataFrame, body_size: float = 20.0) -> dict:
    n_rows = int(len(df))
    out = {
        "n_rows": n_rows,
        "n_tracks": int(df["TrajectoryID"].nunique()) if n_rows else 0,
        "realtime_flips_per_1k": 0.0,
        "final_label_switches": 0,
        "teleports_per_1k": 0.0,
        "unknown_fraction": 0.0,
    }
    if n_rows == 0:
        return out
    flips = teleports = 0
    for _, grp in df.groupby("TrajectoryID", sort=False):
        g = grp.sort_values("FrameID")
        if "IdentityRealtimeLabel" in g.columns:
            flips += _label_switches(g["IdentityRealtimeLabel"])
        if "IdentityFinalLabel" in g.columns:
            out["final_label_switches"] += _label_switches(g["IdentityFinalLabel"])
        xy = g[["X", "Y"]].to_numpy(dtype=float)
        fr = g["FrameID"].to_numpy(dtype=float)
        if len(xy) >= 2:
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            dfr = np.clip(np.diff(fr), 1, None)
            teleports += int(np.sum((step / dfr) > 5.0 * body_size))
    out["realtime_flips_per_1k"] = round(1000.0 * flips / n_rows, 3)
    out["teleports_per_1k"] = round(1000.0 * teleports / n_rows, 3)
    if "IdentityRealtimeLabel" in df.columns:
        lbl = df["IdentityRealtimeLabel"].astype(str).str.strip().str.lower()
        out["unknown_fraction"] = round(float(lbl.isin(_UNKNOWN).mean()), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--body-size", type=float, default=20.0)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    print(json.dumps(compute_identity_metrics(df, args.body_size), indent=2))


if __name__ == "__main__":
    main()
```

Also create empty `tools/identity_bench/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass** — same command as Step 2, expected PASS. (Note the test imports `tools.identity_bench...` — the `$PWD` entry on PYTHONPATH makes `tools` importable.)

- [ ] **Step 5: Commit**

```bash
git add tools/identity_bench/ tests/identity/test_identity_metrics.py
git commit -m "feat(identity-bench): fixture-level identity retention metrics CLI"
```

---

### Task 3: Fix the slot-lock sign (audit fix 1, F3)

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/online.py` (`_apply_slot_lock_bias` at :354-365, `update_frame` step-3 loop at :437-440, `_solve_visible_assignment` at :498-522)
- Test: `tests/identity/test_retention_benchmark.py` (remove one xfail)

**Interfaces:**
- Produces: `OnlineIdentityDecoder._assignment_probs(belief) -> np.ndarray` — the lock-biased posterior used ONLY for the uniqueness assignment; the stored `belief.log_posterior` is never mutated by the lock anymore. `_apply_slot_lock_bias` is deleted.

Rationale (from audit F3): `log_bias = log(0.9) < 0` *penalizes* the locked label, and it mutates the persistent belief every frame (compounding). Replace with a non-persistent log-odds bonus applied only at assignment time: `bonus = log((1+s)/(1-s))` (s=0.9 → +2.94 nats — real hysteresis, but overridable by sustained contrary evidence and by the swap/revision paths, which all use raw posteriors).

- [ ] **Step 1: Remove the xfail marker** from `test_target_slot_lock_bias_boosts_locked_label` in `tests/identity/test_retention_benchmark.py`.

- [ ] **Step 2: Run it to verify it fails** — `PYTHONPATH=$PWD/src conda run -n hydra-mps python -m pytest tests/identity/test_retention_benchmark.py::test_target_slot_lock_bias_boosts_locked_label -v` — expected FAIL (biased < raw).

- [ ] **Step 3: Implement**

In `online.py`, DELETE `_apply_slot_lock_bias` (lines 354-365) and ADD:

```python
    def _assignment_probs(self, belief: TrackIdentityBelief) -> np.ndarray:
        """Posterior used for the uniqueness assignment: the raw posterior
        plus a NON-PERSISTENT log-odds bonus on the slot-locked label.

        The stored belief is never mutated here — the lock is assignment
        hysteresis, not evidence (audit F3: the old in-place log(s) 'bias'
        both had the wrong sign and compounded across frames)."""
        probs = self._posterior_probs(belief)
        label = belief.slot_lock_label
        s = float(np.clip(belief.slot_lock_strength, 0.0, 0.999))
        if not label or s <= 0.0:
            return probs
        try:
            lock_idx = self._catalog.index_of(label)
        except KeyError:
            return probs
        log_p = np.log(np.clip(probs, 1e-300, None))
        log_p[lock_idx] += np.log((1.0 + s) / (1.0 - s))
        log_p -= np.logaddexp.reduce(log_p)
        return np.exp(log_p)
```

In `update_frame`, DELETE the step-3 loop:

```python
        # Step 3: apply lock bias (post-swap, so the bias follows the new
        # committed identity)
        for slot in visible_slots:
            self._apply_slot_lock_bias(self._beliefs[slot])
```

(renumber the comment for step 4 accordingly). In `_solve_visible_assignment`, change:

```python
        posterior_probs = [
            self._posterior_probs(self._beliefs[slot]) for slot in visible_slots
        ]
```

to:

```python
        posterior_probs = [
            self._assignment_probs(self._beliefs[slot]) for slot in visible_slots
        ]
```

Also update the module docstring line for `IDENTITY_SLOT_LOCK_STRENGTH` ("soft-lock bias weight" → "assignment-time log-odds hysteresis strength") and the `slot_lock_strength` field docstring in `TrackIdentityBelief`.

- [ ] **Step 4: Check for other callers** — `grep -rn "_apply_slot_lock_bias" src/ tests/` must return nothing (the only known external user was a session scratchpad probe). If a test references it, update that test to use `_assignment_probs`.

- [ ] **Step 5: Run the full retention suite** — `PYTHONPATH=$PWD/src conda run -n hydra-mps python -m pytest tests/identity/test_retention_benchmark.py tests/identity/test_substrate.py -v` — expected: previous invariants still pass, lock target now passes, remaining 5 targets still xfail.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/individual/identity/online.py tests/identity/test_retention_benchmark.py
git commit -m "fix(identity): slot-lock bias — non-persistent assignment-time log-odds bonus (audit F3)"
```

---

### Task 4: Emit every hardcoded decoder knob from the schema (audit fix 12)

**Files:**
- Modify: `src/hydra_suite/trackerkit/config/identity_schema.py`
- Modify: `src/hydra_suite/trackerkit/engine_params.py` (identity emission block at :1113-1146)
- Test: `tests/identity/test_identity_config_schema.py` (extend)

**Interfaces:**
- Produces engine keys (consumed by Tasks 5-10): `IDENTITY_PER_FRAME_EVIDENCE_CAP` (float, ≤0 = off, default **1.0**), `IDENTITY_PROB_FLOOR` (default **1e-3**), `IDENTITY_EVIDENCE_TAU` (default **5.0**), `IDENTITY_COMMIT_MIN_HITS` (5), `IDENTITY_COMMIT_REVISION_MIN_FRAMES` (8), `IDENTITY_SLOT_LOCK_MIN_FRAMES` (30), `IDENTITY_SLOT_LOCK_STRENGTH` (0.9), `IDENTITY_SLOT_LOCK_OVERRIDE_MARGIN` (0.5), `IDENTITY_RESPAWN_PRIOR_STRENGTH` (0.75), `IDENTITY_RESPAWN_PRIOR_DECAY` (0.97), `IDENTITY_RESPAWN_PRIOR_MAX_GAP` (120), `IDENTITY_REJOIN_CONFIRM_FRAMES` (3), `IDENTITY_SPLIT_ON_REALTIME_SWITCH` (True).
- Note: this task only PLUMBS the keys. Consumers that don't read a key yet simply ignore it — behavior changes land in the later tasks that read them (5, 6, 7, 9, 10, 14).

- [ ] **Step 1: Write the failing test** (append to `tests/identity/test_identity_config_schema.py`, following that file's existing style for building a config and calling `IdentityConfig.from_engine_config` / `build_engine_params` — read the file first and reuse its fixtures):

```python
def test_new_identity_knobs_have_defaults_and_roundtrip():
    cfg = IdentityConfig()
    assert cfg.robustness.per_frame_evidence_cap == 1.0
    assert cfg.robustness.prob_floor == 1e-3
    assert cfg.robustness.evidence_tau == 5.0
    assert cfg.realtime.commit_min_hits == 5
    assert cfg.realtime.commit_revision_min_frames == 8
    assert cfg.realtime.respawn_prior_max_gap == 120
    assert cfg.realtime.slot_lock.min_frames == 30
    assert cfg.realtime.slot_lock.strength == 0.9
    assert cfg.realtime.slot_lock.override_margin == 0.5
    assert cfg.realtime.slot_lock.rejoin_confirm_frames == 3
    assert cfg.posthoc.split_on_realtime_switch is True
    assert IdentityConfig.from_dict(cfg.to_dict()) == cfg
```

Plus an engine-params emission test (reuse the file's existing pattern for asserting emitted keys) checking every key/default in the Interfaces list above appears in the built params dict.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement schema fields**

`SlotLockConfig` gains: `min_frames: int = 30`, `strength: float = 0.9`, `override_margin: float = 0.5`, `rejoin_confirm_frames: int = 3`. `RealtimeIdentityConfig` gains: `commit_min_hits: int = 5`, `commit_revision_min_frames: int = 8`, `respawn_prior_strength: float = 0.75`, `respawn_prior_decay: float = 0.97`, `respawn_prior_max_gap: int = 120`. `RobustnessConfig` becomes active (delete the "Reserved (Phase 3)" comment): `per_frame_evidence_cap: float = 1.0`, `prob_floor: float = 1e-3`, plus new `evidence_tau: float = 5.0`. `PostHocIdentityConfig` gains `split_on_realtime_switch: bool = True`. In `from_engine_config`, read each from the persisted config with the same defaults (`cfg_get(cfg, "identity_per_frame_evidence_cap", 1.0)` etc., slot-lock extras from `advanced.get(...)` like the existing slot-lock fields) and pass `robustness=RobustnessConfig(...)` into the constructed `cls(...)` (it is currently omitted).

- [ ] **Step 4: Implement emission** — in the `engine_params.py` identity block (after `"IDENTITY_REJOIN_DIST_FLOOR": ...`), add one line per key in the Interfaces list, sourced from `identity_cfg.robustness.*`, `identity_cfg.realtime.*`, `identity_cfg.realtime.slot_lock.*`, `identity_cfg.posthoc.split_on_realtime_switch`.

- [ ] **Step 5: Run tests** — the new tests plus the whole existing file: `PYTHONPATH=$PWD/src conda run -n hydra-mps python -m pytest tests/identity/test_identity_config_schema.py -v`. Existing assertions about `RobustnessConfig` defaults (0.0/0.0) will fail — update them to the new active defaults; that default change is this task's intent. Also run `tests/test_engine_params*.py` if present (`ls tests/ | grep engine_params`) and fix any golden-key-list assertions.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/trackerkit/config/identity_schema.py src/hydra_suite/trackerkit/engine_params.py tests/identity/test_identity_config_schema.py
git commit -m "feat(identity): emit robustness/commit/lock/respawn/rejoin knobs from the schema (audit fix 12)"
```

---

### Task 5: Wire the robustness cap/floor into the online decoder (audit fix 2, F1+F2)

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/online.py` (`__init__` config block :158-192, `_fuse_evidence` :334-352, module docstring key list)
- Test: `tests/identity/test_retention_benchmark.py` (no xfail flips yet — this task's scenario gate is quantitative), plus a new direct unit test in `tests/identity/test_online_robustness.py`

**Interfaces:**
- Consumes: engine keys from Task 4; `substrate.fuse_log_evidence(per_frame_cap=..., prob_floor=...)` (already implemented + tested in `tests/identity/test_substrate.py`).
- Produces: decoder attributes `_per_frame_evidence_cap` (float, `inf` when key ≤ 0) and `_prob_floor` (float ≥ 0), read by Task 7's tempering code.

- [ ] **Step 1: Write the failing test**

```python
# tests/identity/test_online_robustness.py
"""Cap/floor wiring: the decoder must pass the robustness knobs to
substrate.fuse_log_evidence (audit F1/F2 — they exist but were unplugged)."""
import numpy as np

from tests.identity.retention_scenarios import drive_constant, make_decoder


def test_prob_floor_keeps_all_hypotheses_reachable():
    dec = make_decoder({"IDENTITY_PROB_FLOOR": 1e-3, "IDENTITY_PER_FRAME_EVIDENCE_CAP": 1.0})
    hist = drive_constant(dec, 0, "A", 0.9, 50)
    bel = dec.get_belief(0)
    probs = dec._posterior_probs(bel)
    assert float(probs.min()) >= 1e-3 * 0.99  # floor holds after 50 fusions


def test_cap_slows_single_streak_takeover():
    slow = make_decoder({"IDENTITY_PER_FRAME_EVIDENCE_CAP": 1.0, "IDENTITY_PROB_FLOOR": 1e-3})
    fast = make_decoder({"IDENTITY_PER_FRAME_EVIDENCE_CAP": 0.0, "IDENTITY_PROB_FLOOR": 0.0})
    h_slow = drive_constant(slow, 0, "A", 0.9, 10)
    h_fast = drive_constant(fast, 0, "A", 0.9, 10)
    # same 10 frames: capped belief must be strictly less extreme
    assert h_slow[-1]["p"]["A"] < h_fast[-1]["p"]["A"]


def test_zero_values_reproduce_legacy_behavior():
    legacy = make_decoder({"IDENTITY_PER_FRAME_EVIDENCE_CAP": 0.0, "IDENTITY_PROB_FLOOR": 0.0,
                           "IDENTITY_EVIDENCE_TAU": 1.0})
    h = drive_constant(legacy, 0, "A", 0.6, 12)
    # audit Appendix A measured: commit in <= 5 frames at 0.6 with legacy fusion
    assert any(r["committed"] for r in h[:6])
```

- [ ] **Step 2: Run to verify the first two fail** (decoder ignores the keys today; floor test fails because min prob is ~1e-30).

- [ ] **Step 3: Implement** — in `__init__` (after the swap params):

```python
        _cap_raw = float(params.get("IDENTITY_PER_FRAME_EVIDENCE_CAP", 1.0))
        self._per_frame_evidence_cap: float = _cap_raw if _cap_raw > 0.0 else float("inf")
        self._prob_floor: float = max(0.0, float(params.get("IDENTITY_PROB_FLOOR", 1e-3)))
```

In `_fuse_evidence`, change the fuse call to:

```python
            belief.log_posterior = substrate.fuse_log_evidence(
                belief.log_posterior,
                ev.log_probs,
                per_frame_cap=self._per_frame_evidence_cap,
                prob_floor=self._prob_floor,
            )
```

Add the two keys (with defaults and the `<=0 = off` convention) to the module docstring's configuration-keys table.

- [ ] **Step 4: Run tests** — new file + retention benchmark + `tests/identity/test_substrate.py`. Expected: new tests pass; invariants pass. `test_target_moderate_wrong_streak_does_not_flip` may STILL xfail (cap alone doesn't stop a 12-frame streak — Task 7 finishes it); if it XPASSes, remove its xfail marker now and note in the commit message that cap+floor alone sufficed, then skip the marker-removal step in Task 7.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/identity/online.py tests/identity/test_online_robustness.py
git commit -m "fix(identity): wire per-frame evidence cap + prob floor into online fusion (audit F1/F2)"
```

---

### Task 6: Commit revision requires sustained counter-evidence (audit fix 5, F4)

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/online.py` (`TrackIdentityBelief` fields :78-91, `__init__` config block, `_update_commitment` :524-618)
- Test: `tests/identity/test_retention_benchmark.py` (remove `test_target_short_strong_wrong_burst_does_not_revise` xfail)

**Interfaces:**
- Consumes: `IDENTITY_COMMIT_REVISION_MIN_FRAMES` (Task 4, default 8).
- Produces: belief fields `revision_candidate: Optional[str]`, `revision_streak: int` (Task 10's TTL code coexists with these; the swap path `_execute_swap` is untouched and remains the paired-swap mechanism).

- [ ] **Step 1: Remove the xfail marker** from `test_target_short_strong_wrong_burst_does_not_revise`.

- [ ] **Step 2: Run it to verify it fails** (today a 5-frame 0.9 burst revises the commitment).

- [ ] **Step 3: Implement**

Add to `TrackIdentityBelief`:

```python
    revision_candidate: Optional[str] = None
    revision_streak: int = 0
```

Add to `__init__`:

```python
        self._commit_revision_min_frames: int = max(
            1, int(params.get("IDENTITY_COMMIT_REVISION_MIN_FRAMES", 8))
        )
```

In `_update_commitment`: (a) in the early-return branch where the commit conditions are NOT met (`if not (label and confidence >= ... )`), also reset the revision counter before returning:

```python
            belief.stable_count = 0
            belief.revision_candidate = None
            belief.revision_streak = 0
            return
```

(b) replace the committed-revision branch (currently `if belief.committed and belief.committed_label not in (None, label): ... if confidence - committed_conf < self._slot_lock_override_margin: ...`) with:

```python
        if belief.committed and belief.committed_label not in (None, label):
            try:
                committed_conf = float(probs[belief.committed_index])
            except IndexError:
                committed_conf = 0.0
            if belief.revision_candidate == label:
                belief.revision_streak += 1
            else:
                belief.revision_candidate = label
                belief.revision_streak = 1
            if (
                belief.revision_streak < self._commit_revision_min_frames
                or confidence - committed_conf < self._slot_lock_override_margin
            ):
                belief.stable_count = 0
                return
            log.debug(
                "Slot %d revised commitment '%s' -> '%s' after %d sustained frames "
                "(override margin %.3f)",
                belief.slot_index,
                belief.committed_label,
                label,
                belief.revision_streak,
                confidence - committed_conf,
            )
            belief.slot_lock_label = None
            belief.slot_lock_strength = 0.0
            belief.slot_lock_frame = 0
            belief.revision_candidate = None
            belief.revision_streak = 0
```

(c) directly after that branch, reset the counter when the committed label is re-confirmed:

```python
        if belief.committed and belief.committed_label == label:
            belief.revision_candidate = None
            belief.revision_streak = 0
```

- [ ] **Step 4: Run the retention suite** — target passes; ALL invariants must still pass, especially `test_invariant_sustained_wrong_evidence_eventually_corrects` (150-frame streak > 8-frame debounce) and `test_invariant_true_swap_is_corrected` (swap path bypasses `_update_commitment`, must be unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/identity/online.py tests/identity/test_retention_benchmark.py
git commit -m "fix(identity): commitment revision requires sustained counter-evidence (audit F4)"
```

---

### Task 7: Temper evidence by effective sample size (audit fix 3, F1)

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/online.py` (`__init__`, `_fuse_evidence`, docstring)
- Create: `tools/identity_bench/estimate_tau.py`
- Test: `tests/identity/test_retention_benchmark.py` (remove `test_target_moderate_wrong_streak_does_not_flip` xfail, unless Task 5 already did), `tests/identity/test_online_robustness.py` (extend), `tests/identity/test_estimate_tau.py`

**Interfaces:**
- Consumes: `IDENTITY_EVIDENCE_TAU` (Task 4, default 5.0; 1.0 = off).
- Produces: decoder attribute `_evidence_tau`; `estimate_tau_from_series(top1_probs: np.ndarray) -> float` in `tools/identity_bench/estimate_tau.py` (integrated autocorrelation time; Task 13 reuses the same τ semantics offline).

- [ ] **Step 1: Write the failing tests**

Remove the xfail from `test_target_moderate_wrong_streak_does_not_flip` (if still marked). Append to `tests/identity/test_online_robustness.py`:

```python
def test_tempering_divides_per_frame_information():
    t1 = make_decoder({"IDENTITY_EVIDENCE_TAU": 1.0, "IDENTITY_PER_FRAME_EVIDENCE_CAP": 0.0,
                       "IDENTITY_PROB_FLOOR": 0.0})
    t5 = make_decoder({"IDENTITY_EVIDENCE_TAU": 5.0, "IDENTITY_PER_FRAME_EVIDENCE_CAP": 0.0,
                       "IDENTITY_PROB_FLOOR": 0.0})
    h1 = drive_constant(t1, 0, "A", 0.9, 5)
    h5 = drive_constant(t5, 0, "A", 0.9, 25)
    # 25 tempered frames ~= 5 raw frames of information (within transition-leak slack)
    assert abs(h5[-1]["p"]["A"] - h1[-1]["p"]["A"]) < 0.15
```

And `tests/identity/test_estimate_tau.py`:

```python
import numpy as np

from tools.identity_bench.estimate_tau import estimate_tau_from_series


def test_iid_series_has_tau_near_one():
    rng = np.random.default_rng(0)
    assert estimate_tau_from_series(rng.uniform(size=2000)) < 1.5


def test_correlated_series_has_larger_tau():
    rng = np.random.default_rng(0)
    x = np.zeros(2000)
    for i in range(1, 2000):
        x[i] = 0.9 * x[i - 1] + rng.normal()
    tau = estimate_tau_from_series(x)
    assert tau > 5.0
```

- [ ] **Step 2: Run to verify failures.**

- [ ] **Step 3: Implement decoder tempering** — `__init__`:

```python
        self._evidence_tau: float = max(
            1.0, float(params.get("IDENTITY_EVIDENCE_TAU", 5.0))
        )
```

`_fuse_evidence` (before the fuse call, after the size guard):

```python
            log_probs = ev.log_probs
            if self._evidence_tau > 1.0:
                # Temper: consecutive crops are correlated, not independent
                # confirmations (audit F1). 1/tau is the effective-sample-size
                # correction; cap/floor then bound the tempered contribution.
                log_probs = log_probs / self._evidence_tau
            belief.log_posterior = substrate.fuse_log_evidence(
                belief.log_posterior,
                log_probs,
                per_frame_cap=self._per_frame_evidence_cap,
                prob_floor=self._prob_floor,
            )
```

Document `IDENTITY_EVIDENCE_TAU` in the module docstring.

- [ ] **Step 4: Implement the τ estimator**

```python
# tools/identity_bench/estimate_tau.py
"""Estimate the crop-evidence autocorrelation time tau from a per-frame
top-1 probability series (audit open question 3). tau is the integrated
autocorrelation time: 1 + 2*sum(rho_k) up to the first non-positive rho_k.
Use it to set IDENTITY_EVIDENCE_TAU for a given dataset."""
from __future__ import annotations

import argparse

import numpy as np


def estimate_tau_from_series(x: np.ndarray, max_lag: int = 200) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10 or np.var(x) == 0.0:
        return 1.0
    x = x - x.mean()
    var = float(np.dot(x, x)) / x.size
    tau = 1.0
    for k in range(1, min(max_lag, x.size - 1)):
        rho = float(np.dot(x[:-k], x[k:])) / ((x.size - k) * var)
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return float(tau)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="tracking CSV with an IdentityRealtimeConfidence column")
    ap.add_argument("--column", default="IdentityRealtimeConfidence")
    args = ap.parse_args()
    import pandas as pd

    df = pd.read_csv(args.csv)
    taus = [
        estimate_tau_from_series(g[args.column].to_numpy(dtype=float))
        for _, g in df.groupby("TrajectoryID")
        if len(g) >= 50
    ]
    print(f"per-track tau: median={np.median(taus):.2f} p90={np.percentile(taus, 90):.2f} n={len(taus)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run everything and tune** — retention benchmark + robustness + tau tests. The moderate-streak target (12×0.6 wrong) must now pass AND the invariants must hold — in particular `test_invariant_cold_commit_is_bounded` (≤60 frames at 0.9 with τ=5) and the 150-frame correction invariant. If cold-commit exceeds 60 frames, lower τ toward 3.0 (in BOTH the schema default and the decoder default) rather than weakening the invariant; if the moderate streak still flips, raise `IDENTITY_COMMIT_REVISION_MIN_FRAMES`'s default to 12 (schema + decoder + Task 4 test). Record the final defaults in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/individual/identity/online.py tools/identity_bench/estimate_tau.py tests/identity/test_online_robustness.py tests/identity/test_estimate_tau.py tests/identity/test_retention_benchmark.py
git commit -m "feat(identity): temper correlated evidence by 1/tau + tau estimator tool (audit F1)"
```

---

### Task 8: Give `unknown` a real emission (audit fix 4, F2)

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/catalog.py` (`cnn_log_prior` :200-229)
- Modify: `src/hydra_suite/core/individual/identity/substrate.py` (`_factor_log_prob` :234-288)
- Test: `tests/identity/test_retention_benchmark.py` (remove ambiguity xfail), `tests/identity/test_substrate.py` + `tests/identity/test_evidence_builder_parity.py` + `tests/identity/test_evidence_phase_basis_parity.py` (update expected values)

**Interfaces:**
- Produces: `p(evidence | unknown) = 1/K` (uniform over the model's K mapped classes) instead of the 1e-6 floor, in both the flat CNN mapping and the composite factor mapping. AprilTag's 1e-4 unknown floor is deliberately unchanged (tags are near-deterministic detections; document this in the commit message).

- [ ] **Step 1: Remove the xfail** from `test_target_ambiguous_evidence_keeps_unknown_reachable`; run to confirm it fails.

- [ ] **Step 2: Implement `cnn_log_prior`** — replace `p[0] = floor  # unknown gets the floor` with:

```python
        n_classes = sum(
            1 for lbl in label_map if self.contains(lbl) and lbl != UNKNOWN_LABEL
        )
        # Open-set emission: under the "unknown animal" hypothesis the CNN's
        # output is uninformative, so p(evidence | unknown) = 1/K, not a
        # vanishing floor (audit F2: the 1e-6 floor annihilated the unknown
        # state after a single frame).
        p[0] = max(floor, 1.0 / max(n_classes, 1))
```

- [ ] **Step 3: Implement `_factor_log_prob`** — after the per-branch loops and before `probs /= probs.sum()`, replace the unknown entry (currently left at `floor` from the `np.full` initialization) with:

```python
    n_observed = int(observed[1:].sum())
    probs[0] = max(floor, 1.0 / max(n_observed, 1))
```

- [ ] **Step 4: Run and repair the parity tests** — `PYTHONPATH=$PWD/src conda run -n hydra-mps python -m pytest tests/identity/ -v -x`. `test_substrate.py` / `test_evidence_builder_parity.py` / `test_evidence_phase_basis_parity.py` encode the old floor in expected vectors — update the expected values (the tests' purpose is builder-vs-substrate parity, which is preserved since both sides change together). The ambiguity target must now pass: with conf=0.30 vs uniform 0.25 over K=4, unknown holds ≥0.05. Verify the cold-commit and moderate-streak tests still pass (the unknown emission slightly slows commitment — if cold-commit breaks its 60-frame bound, revisit τ per Task 7 Step 5's rule).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/identity/catalog.py src/hydra_suite/core/individual/identity/substrate.py tests/identity/
git commit -m "fix(identity): unknown state gets a real 1/K emission instead of a 1e-6 floor (audit F2)"
```

---

### Task 9: Harden the identity-first rejoin (audit fix 6, F7)

**Files:**
- Modify: `src/hydra_suite/core/assigners/hungarian.py` (`TrackAssigner.__init__` :111, `_assign_respawn` identity branch :801-868)
- Modify: `src/hydra_suite/trackerkit/config/identity_schema.py` (`rejoin_threshold` default 0.5 → 0.6) and its `from_engine_config` default
- Test: `tests/identity/test_retention_benchmark.py` (remove teleport xfail)

**Interfaces:**
- Consumes: `IDENTITY_REJOIN_CONFIRM_FRAMES` (Task 4, default 3), `IDENTITY_DISPLAY_THRESHOLD`.
- Produces: assigner state `self._rejoin_confirm: dict[int, dict]` and `self._respawn_tick: int`; √t motion budget; rejoin threshold clamped to ≥ display threshold. Task 10's helper uses the same √t budget law.

- [ ] **Step 1: Remove the xfail** from `test_target_far_single_frame_claim_does_not_rejoin`; run to confirm it fails (today: linear budget 600px admits the 400px jump and one 0.55 frame executes it). Confirm `test_invariant_near_supported_claim_rejoins` currently passes (it uses `streak_calls=3` precisely so it stays valid after this task).

- [ ] **Step 2: Implement**

In `TrackAssigner.__init__` add:

```python
        # Identity-rejoin confirmation state (audit F7): slot -> streak info.
        self._rejoin_confirm: dict = {}
        self._respawn_tick: int = 0
```

In `_assign_respawn`, at the top of the identity branch (`if committed_lost and association_data:`), add `self._respawn_tick += 1` and clamp the threshold:

```python
            rejoin_threshold = float(p.get("IDENTITY_REJOIN_THRESHOLD", 0.6))
            rejoin_threshold = max(
                rejoin_threshold, float(p.get("IDENTITY_DISPLAY_THRESHOLD", 0.6))
            )
```

Replace the linear budget in `_within_budget` (keep the `missed_frames is None` early return):

```python
                lost_n = max(int(missed_frames[slot_idx]), 1)
                # sqrt(t) diffusion growth, not linear flight (audit F7): an
                # unobserved animal's plausible displacement grows as a random
                # walk, so long occlusions no longer license arena-scale jumps.
                budget = max(
                    budget_floor,
                    v_max_per_frame * budget_safety * math.sqrt(lost_n),
                )
```

(check `import math` exists at the top of hungarian.py; add it if absent). Then replace the immediate pairing loop:

```python
            for det_j, (score, slot) in det_best.items():
                identity_rejoin_pairs.append((slot, det_j))
                identity_claimed_dets.add(det_j)
```

with a confirmation window (the same candidate must win on consecutive frames at spatially coherent positions):

```python
            confirm_n = max(1, int(p.get("IDENTITY_REJOIN_CONFIRM_FRAMES", 3)))
            tick = self._respawn_tick
            for det_j, (score, slot) in det_best.items():
                det_xy = np.asarray(meas[det_j][:2], dtype=np.float64)
                st = self._rejoin_confirm.get(slot)
                if (
                    st is not None
                    and tick - st["tick"] == 1
                    and float(np.linalg.norm(det_xy - st["xy"])) <= v_max_per_frame
                ):
                    streak = st["streak"] + 1
                else:
                    streak = 1
                self._rejoin_confirm[slot] = {"tick": tick, "xy": det_xy, "streak": streak}
                if streak >= confirm_n:
                    identity_rejoin_pairs.append((slot, det_j))
                    identity_claimed_dets.add(det_j)
                    self._rejoin_confirm.pop(slot, None)
            # Streaks are consecutive: drop entries not renewed this frame.
            self._rejoin_confirm = {
                s: st for s, st in self._rejoin_confirm.items() if st["tick"] == tick
            }
```

Also update the default in the engine-key read to match the schema change (`IDENTITY_REJOIN_THRESHOLD` default 0.6) and change `RealtimeIdentityConfig.rejoin_threshold` default + `from_engine_config`'s `cfg_get(cfg, "identity_rejoin_threshold", 0.5)` to 0.6.

- [ ] **Step 3: Run** the retention benchmark. Teleport target passes (single far frame rejected on all three grounds: budget, threshold, confirmation); near-rejoin invariant passes (3 coherent calls at 60px). Also run `PYTHONPATH=$PWD/src conda run -n hydra-mps python -m pytest tests/core -k "hungarian or assign" -v` (existing assigner tests; if any legacy test asserts single-frame rejoin, update it to call the phase 3× and cite audit F7 in a comment).

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/core/assigners/hungarian.py src/hydra_suite/trackerkit/config/identity_schema.py tests/
git commit -m "fix(identity): rejoin needs confirmation window + sqrt(t) budget + display-level threshold (audit F7)"
```

---

### Task 10: Respawn prior keyed on space + committed-lost TTL (audit fix 13, F9)

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/online.py` (`decay_absent_slot_beliefs` :822-833, `update_frame` visible loop, new module-level helper, `TrackIdentityBelief` field)
- Modify: `src/hydra_suite/core/tracking/worker.py` (every `clear_slot(` call site that passes `respawn_frame_idx`)
- Test: `tests/identity/test_retention_benchmark.py` (remove TTL xfail), `tests/identity/test_online_robustness.py` (extend)

**Interfaces:**
- Produces: module-level `respawn_carry_within_budget(last_xy, spawn_xy, gap_frames, params) -> bool` in `online.py`; belief field `absent_streak: int = 0`. The worker gates prior-carry by calling the helper and passing `respawn_frame_idx=None` when out of budget — no decoder API change for the spatial half.

- [ ] **Step 1: Remove the xfail** from `test_target_committed_lost_slot_expires` and add the helper test to `tests/identity/test_online_robustness.py`:

```python
def test_respawn_carry_budget_is_spatial():
    from hydra_suite.core.individual.identity.online import respawn_carry_within_budget

    params = {"REFERENCE_BODY_SIZE": 20.0, "RESIZE_FACTOR": 1.0,
              "KALMAN_MAX_VELOCITY_MULTIPLIER": 2.0, "IDENTITY_REJOIN_VELOCITY_BUDGET": 1.5}
    # sqrt(10)*40*1.5 ~= 190px budget
    assert respawn_carry_within_budget((0.0, 0.0), (100.0, 0.0), 10, params)
    assert not respawn_carry_within_budget((0.0, 0.0), (400.0, 0.0), 10, params)
    assert respawn_carry_within_budget(None, (400.0, 0.0), 10, params)  # unknown -> permissive
```

Run both; expect FAIL (helper missing; TTL absent).

- [ ] **Step 2: Implement in `online.py`**

Module-level helper (after the dataclasses):

```python
def respawn_carry_within_budget(
    last_xy: Optional[tuple[float, float]],
    spawn_xy: Optional[tuple[float, float]],
    gap_frames: int,
    params: dict[str, Any],
) -> bool:
    """Should a lost slot's identity prior be carried into a respawn at
    spawn_xy? Same sqrt(t) diffusion law as the rejoin budget (audit F9:
    the prior was slot-index-keyed, teleporting identities across space)."""
    if last_xy is None or spawn_xy is None:
        return True
    body = float(params.get("REFERENCE_BODY_SIZE", 20.0)) * float(
        params.get("RESIZE_FACTOR", 1.0)
    )
    v_max = float(params.get("KALMAN_MAX_VELOCITY_MULTIPLIER", 2.0)) * body
    safety = float(params.get("IDENTITY_REJOIN_VELOCITY_BUDGET", 1.5))
    budget = max(2.0 * body, v_max * safety * math.sqrt(max(int(gap_frames), 1)))
    dist = math.hypot(
        float(spawn_xy[0]) - float(last_xy[0]), float(spawn_xy[1]) - float(last_xy[1])
    )
    return dist <= budget
```

(add `import math` to online.py). Add `absent_streak: int = 0` to `TrackIdentityBelief`. In `update_frame`'s per-slot loop (after `belief = self._get_or_create_belief(...)`), add `belief.absent_streak = 0`. Replace `decay_absent_slot_beliefs`'s body:

```python
        for slot in absent_slots:
            belief = self._beliefs.get(slot)
            if belief is None or not belief.committed_label:
                continue
            self._predict_belief(belief)
            belief.absent_streak += 1
            if belief.absent_streak > self._respawn_prior_max_gap:
                log.info(
                    "Slot %d committed identity '%s' expired after %d absent frames",
                    slot,
                    belief.committed_label,
                    belief.absent_streak,
                )
                belief.committed = False
                belief.committed_label = None
                belief.committed_index = 0
                belief.slot_lock_label = None
                belief.slot_lock_strength = 0.0
                belief.absent_streak = 0
```

- [ ] **Step 3: Wire the worker's spatial gate** — `grep -n "clear_slot(" src/hydra_suite/core/tracking/worker.py`. At each call that passes `respawn_frame_idx=...` (the respawn/slot-reuse sites around :3132-3145 and :3384-3389), the worker knows the slot's last KF position (`kf_manager.X[slot, :2]` BEFORE the reset) and the claiming detection's position; wrap:

```python
                    _carry_ok = online_identity.respawn_carry_within_budget(
                        tuple(_last_xy), tuple(_spawn_xy), int(_gap_frames), p
                    )
                    _identity_online_decoder.clear_slot(
                        slot,
                        reason=...,
                        respawn_frame_idx=(frame_idx if _carry_ok else None),
                    )
```

using the surrounding block's actual variable names for slot/positions/gap (read 30 lines of context at each site first; `_gap_frames` is the slot's missed-frame count available at those sites). Import the helper where the decoder is imported (:1827-1830).

- [ ] **Step 4: Run** the retention suite + robustness file (all pass) and a worker smoke: `PYTHONPATH=$PWD/src conda run -n hydra-mps python -c "import hydra_suite.core.tracking.worker"`.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/identity/online.py src/hydra_suite/core/tracking/worker.py tests/identity/
git commit -m "fix(identity): spatially gate respawn-prior carry + expire committed-lost beliefs (audit F9)"
```

---

### Task 11: One likelihood for association, rejoin, and fusion (audit fix 7, F6 — consistency half)

**Files:**
- Create: `src/hydra_suite/core/individual/identity/association_evidence.py`
- Modify: `src/hydra_suite/core/tracking/worker.py` (association block :2692-2760 and the post-assignment evidence-fusion block ~:2966-3072)
- Modify: `src/hydra_suite/trackerkit/config/identity_schema.py` + `src/hydra_suite/core/assigners/hungarian.py` (align `ASSOCIATION_IDENTITY_HINT_SCALE` defaults)
- Test: `tests/identity/test_association_evidence.py`

**Interfaces:**
- Produces: `build_detection_log_likelihoods(catalog, n_dets, det_tag_ids, tag_label_map, per_det_catalog_log_probs: dict[int, np.ndarray]) -> list[Optional[np.ndarray]]`. The worker computes each detection's calibrated catalog log-prob vector ONCE (the same vectors the post-assignment fusion consumes) and feeds both the association cost/rejoin channel and fusion — eliminating the uncalibrated top-1 twin likelihood (audit F6).
- Default alignment: `RealtimeIdentityConfig.association_weight` default 1.0 → **0.3** (and `from_engine_config`'s `identity_weight` default), matching `hungarian.py:241`'s code fallback. Rationale: until the double-count is fully removed, the weaker coupling is the safe default (audit open question 6).

- [ ] **Step 1: Write the failing test**

```python
# tests/identity/test_association_evidence.py
import numpy as np

from hydra_suite.core.individual.identity.association_evidence import (
    build_detection_log_likelihoods,
)
from tests.identity.retention_scenarios import make_catalog


def test_combines_tag_and_calibrated_cnn():
    cat = make_catalog()
    cal = np.log(np.array([0.05, 0.8, 0.05, 0.05, 0.05]))
    out = build_detection_log_likelihoods(
        catalog=cat,
        n_dets=3,
        det_tag_ids=[-1, 7, -1],
        tag_label_map={7: "B"},
        per_det_catalog_log_probs={0: cal, 1: cal},
    )
    assert len(out) == 3
    assert np.allclose(out[0], cal)                      # CNN only
    assert out[2] is None                                # no evidence at all
    tag_only = cat.apriltag_log_prior(7, {7: "B"})
    assert np.allclose(out[1], tag_only + cal)           # tag + CNN summed
```

Run: fails with ModuleNotFoundError.

- [ ] **Step 2: Implement the module**

```python
# src/hydra_suite/core/individual/identity/association_evidence.py
"""Shared per-detection identity log-likelihood builder.

One likelihood for all three realtime consumers — the Bayesian association
cost, the identity-first rejoin score, and (upstream) belief fusion — built
from the SAME calibrated catalog log-prob vectors the fusion path uses.
Replaces the worker's inline uncalibrated top-1 reconstruction (audit F6:
twin likelihoods made the cost/rejoin channel disagree with fusion)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from hydra_suite.core.individual.identity.catalog import IdentityCatalog


def build_detection_log_likelihoods(
    catalog: IdentityCatalog,
    n_dets: int,
    det_tag_ids: list[int],
    tag_label_map: dict[int, str],
    per_det_catalog_log_probs: dict[int, np.ndarray],
) -> list[Optional[np.ndarray]]:
    out: list[Optional[np.ndarray]] = []
    for j in range(n_dets):
        ll: Optional[np.ndarray] = None
        tid = int(det_tag_ids[j]) if j < len(det_tag_ids) else -1
        if tid >= 0:
            ll = catalog.apriltag_log_prior(tid, tag_label_map)
        cal = per_det_catalog_log_probs.get(j)
        if cal is not None:
            cal = np.asarray(cal, dtype=np.float64)
            ll = cal if ll is None else ll + cal
        out.append(ll)
    return out
```

- [ ] **Step 3: Rewire the worker** — read `src/hydra_suite/core/tracking/worker.py:2960-3080` to find where the post-assignment fusion builds each matched detection's calibrated catalog log-prob vector (the `IdentityEvidence` construction from `_cnn_frame_preds_all` / the evidence-builder remap). Hoist the per-detection calibrated-vector computation ABOVE the association block into a dict `_det_catalog_log_probs: dict[int, np.ndarray]` (keyed by detection index, identity-providing CNN phases only, summed in log-space across phases exactly as the fusion path does), then: (a) replace the whole inline top-1 reconstruction block at :2705-2747 with one call to `build_detection_log_likelihoods(_cat, _n_dets, _det_tag_ids, _tag_label_map, _det_catalog_log_probs)`; (b) make the fusion block consume `_det_catalog_log_probs` for its CNN vectors instead of recomputing. Preserve the existing `try/except` non-fatal guard. If the fusion path's vectors are only computable per-matched-slot (not per-detection) — check first — compute per-detection unconditionally (evidence exists per detection before matching) and index by detection id at both sites.

- [ ] **Step 4: Align the hint-scale defaults** — `identity_schema.py`: `association_weight: float = 0.3` and `from_engine_config`: `cfg_get(cfg, "identity_weight", 0.3)`. Verify `hungarian.py`'s fallback is `0.3` (grep `ASSOCIATION_IDENTITY_HINT_SCALE`); update the schema-defaults test from Task 4/existing tests.

- [ ] **Step 5: Run** — new test, retention suite, schema tests, plus worker import smoke. Then run the two identity fixture clips end-to-end once (see Task 16 Step 1 for the command) and confirm `identity_metrics.py` output is not worse than the pre-Task-11 run (keep the JSON outputs; the audit expects same-or-fewer flips since cost and fusion now agree).

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/individual/identity/association_evidence.py src/hydra_suite/core/tracking/worker.py src/hydra_suite/trackerkit/config/identity_schema.py src/hydra_suite/core/assigners/hungarian.py tests/identity/test_association_evidence.py
git commit -m "fix(identity): one calibrated likelihood for association/rejoin/fusion; align hint-scale default (audit F6)"
```

---

### Task 12: Point the solver's online prior at IdentityRealtime* (audit fix 8, F12)

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/offline.py` (`_LABEL_COL`/`_CONF_COL` :35-36, `_build_traj_summaries` OnlineLabel block :973-996)
- Test: `tests/identity/test_offline_online_prior.py`

**Interfaces:**
- Produces: `_build_traj_summaries` reads `OnlineLabel`/`OnlineConfidence` from `C.REALTIME_LABEL`/`C.REALTIME_CONFIDENCE` (rows where `C.REALTIME_COMMITTED` is truthy, when that column exists), and NEVER from `C.FINAL_LABEL` (audit F12: the 0.25-weight online prior read a column that doesn't exist yet on first pass and is a stale previous offline answer on re-runs).
- Keep `_LABEL_COL = C.FINAL_LABEL` for any WRITE-side uses — first `grep -n "_LABEL_COL\|_CONF_COL" src/hydra_suite/core/individual/identity/offline.py` and change only the summary-reading site; add `_ONLINE_LABEL_COL = C.REALTIME_LABEL`, `_ONLINE_CONF_COL = C.REALTIME_CONFIDENCE`, `_ONLINE_COMMITTED_COL = C.REALTIME_COMMITTED`.

- [ ] **Step 1: Write the failing test**

```python
# tests/identity/test_offline_online_prior.py
"""The fragment solver's online prior must come from the realtime columns,
never from a previous offline pass's IdentityFinalLabel (audit F12)."""
import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.identity.offline import _build_traj_summaries
from tests.identity.retention_scenarios import make_catalog


def _df(realtime, final, committed=True):
    n = 20
    return pd.DataFrame(
        {
            "TrajectoryID": [1] * n,
            "FrameID": range(n),
            "X": np.zeros(n),
            "Y": np.zeros(n),
            C.REALTIME_LABEL: [realtime] * n,
            C.REALTIME_CONFIDENCE: [0.9] * n,
            C.REALTIME_COMMITTED: [committed] * n,
            C.FINAL_LABEL: [final] * n,
            C.FINAL_CONFIDENCE: [0.99] * n,
        }
    )


def test_online_prior_reads_realtime_not_final():
    summaries = _build_traj_summaries(_df(realtime="A", final="B"), make_catalog())
    assert summaries.iloc[0]["OnlineLabel"] == "A"


def test_uncommitted_realtime_rows_do_not_form_a_prior():
    summaries = _build_traj_summaries(_df(realtime="A", final="B", committed=False),
                                      make_catalog())
    assert summaries.iloc[0]["OnlineLabel"] == "unknown"
```

Run: first test fails (reads FINAL_LABEL → "B").

- [ ] **Step 2: Implement** — in `_build_traj_summaries`, replace the `label_col = grp_sorted.get(_LABEL_COL, ...)` block's source columns with the realtime ones and filter by committed:

```python
        label_col = grp_sorted.get(
            _ONLINE_LABEL_COL,
            pd.Series("unknown", index=grp_sorted.index, dtype=object),
        )
        if _ONLINE_COMMITTED_COL in grp_sorted.columns:
            committed_mask = (
                grp_sorted[_ONLINE_COMMITTED_COL]
                .astype(str).str.strip().str.lower()
                .isin({"true", "1", "1.0", "yes"})
            )
            label_col = label_col.where(committed_mask, other="unknown")
```

and use `_ONLINE_CONF_COL` where the block reads `_CONF_COL`. Update the function docstring's "OnlineLabel/OnlineConfidence are always read from IdentityFinalLabel..." paragraph to describe the realtime source and cite audit F12.

- [ ] **Step 3: Run** — new test + `tests/test_fragment_solver.py` + `tests/identity/test_honesty_fix.py` (fix any tests that constructed FINAL_LABEL-based online priors: point them at the realtime columns — that redirection is this task's intent).

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/core/individual/identity/offline.py tests/
git commit -m "fix(identity): fragment solver online prior reads IdentityRealtime*, never a prior pass's finals (audit F12)"
```

---

### Task 13: Solver support from raw evidence, as a tempered sum (audit fix 9, F13)

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/offline.py` (`run_fragment_solver` :1379-1440, `_evidence_dicts_for_fragment` :850-887, `_build_traj_summaries` signature)
- Test: `tests/identity/test_offline_raw_support.py`, update `tests/test_fragment_solver.py` / `tests/identity/test_offline_evidence_sourcing.py` expectations

**Interfaces:**
- Produces: `run_fragment_solver` keeps BOTH sequences: `raw_evidence` feeds `solve_global_assignment(..., evidence_by_traj=raw_evidence)` (support/stability), smoothed sequences feed ONLY `_annotate_smoothed_labels` and PELT (display + splitting). `_evidence_dicts_for_fragment(known_labels, sequence, evidence_tau=5.0)` computes `CNNLogEvidence[label] = sum(log p_t(label)) / evidence_tau` (a tempered SUM restoring evidence count — audit F13: mean-of-smoothed-logs both double-counted time and discarded count) and `Stability` from the raw sequence. `_build_traj_summaries` gains an `evidence_tau: float = 5.0` keyword threaded from `params.get("IDENTITY_EVIDENCE_TAU", 5.0)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/identity/test_offline_raw_support.py
"""Tempered-sum support: a long consistent fragment must outweigh a short
confident burst (audit F13 — geometric mean discarded evidence count)."""
import numpy as np

from hydra_suite.core.individual.identity.offline import _evidence_dicts_for_fragment


def _seq(label_idx, conf, frames, k=4):
    out = []
    for t in range(frames):
        p = np.full(k + 1, (1.0 - conf) / (k - 1))
        p[0] = 1e-3
        p[label_idx] = conf
        p /= p.sum()
        out.append((t, np.log(p)))
    return out


def test_long_consistent_beats_short_burst():
    known = ["A", "B", "C", "D"]
    _, long_scores, _ = _evidence_dicts_for_fragment(known, _seq(1, 0.6, 100),
                                                     evidence_tau=5.0)
    _, short_scores, _ = _evidence_dicts_for_fragment(known, _seq(2, 0.9, 5),
                                                      evidence_tau=5.0)
    # margin over the runner-up label, per fragment
    long_margin = long_scores["A"] - max(v for k, v in long_scores.items() if k != "A")
    short_margin = short_scores["B"] - max(v for k, v in short_scores.items() if k != "B")
    assert long_margin > short_margin  # count matters again


def test_tau_one_is_plain_sum():
    known = ["A", "B", "C", "D"]
    _, s, _ = _evidence_dicts_for_fragment(known, _seq(1, 0.6, 10), evidence_tau=1.0)
    seq = _seq(1, 0.6, 10)
    expected = float(np.sum([lp[1] for _, lp in seq]))
    assert np.isclose(s["A"], expected)
```

Run: fails (`_evidence_dicts_for_fragment` takes no `evidence_tau`; mean-of-logs makes long_margin ≈ per-frame margin < short burst's).

- [ ] **Step 2: Implement** — `_evidence_dicts_for_fragment(known_labels, sequence, evidence_tau=5.0)`:

```python
    tau = max(1.0, float(evidence_tau))
    cnn_log_scores = {
        label: float(np.sum(known_log[:, idx])) / tau
        for idx, label in enumerate(known_labels)
    }
```

(replace the `np.mean` version; update the docstring paragraph that says the mean convention "is preserved"). Thread `evidence_tau` through `_build_traj_summaries` (new keyword, default 5.0) and from `solve_global_assignment`/callers via `params.get("IDENTITY_EVIDENCE_TAU", 5.0)` — grep for `_build_traj_summaries(` callers and update each. In `run_fragment_solver`, rename `smoothed_by_traj` usage: keep computing the smoothed dict, but pass the RAW dict to the final call:

```python
    assign_evidence = (
        {tid: list(seq) for tid, seq in raw_evidence.items()} if cache is not None and raw_evidence else None
    )
    ...
    return solve_global_assignment(
        split_df, catalog, params, evidence_by_traj=assign_evidence
    )
```

while `_annotate_smoothed_labels` and the PELT block keep receiving `smoothed_by_traj`. Note the `IDENTITY_ENABLE_SMOOTHING=False` branch now only affects PELT/display (raw already drives assignment) — update `run_fragment_solver`'s docstring accordingly.

- [ ] **Step 3: Run and repair** — new test + `tests/test_fragment_solver.py` + `tests/identity/test_offline_evidence_sourcing.py` + `tests/identity/test_offline_smoothing.py`. Tests asserting mean-based `CNNLogEvidence` values or smoothed-input assignment need their expected values recomputed (sum/τ of the same inputs); the direction of every assignment in those tests should be unchanged — if an assignment test's WINNER changes, stop and inspect (that means length now dominates a case the old test deemed evidence-driven; adjust `FRAGMENT_LENGTH_WEIGHT` interplay only with justification written into the test).

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/core/individual/identity/offline.py tests/
git commit -m "fix(identity): offline support = tempered SUM of raw evidence; smoothing only for display/PELT (audit F13)"
```

---

### Task 14: Split fragments at realtime commitment switches (audit fix 10, F14)

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/offline.py` (`run_fragment_solver`, new helper `_realtime_switch_changepoints`)
- Test: `tests/identity/test_offline_realtime_splits.py`

**Interfaces:**
- Consumes: `IDENTITY_SPLIT_ON_REALTIME_SWITCH` (Task 4, default True), `IDENTITY_DISAGREE_MIN_RUN` (existing, default 5), `split_trajectories_at_changepoints` (existing — read its signature/changepoint format in `offline.py` before wiring; it is the same function the PELT branch calls).
- Produces: `_realtime_switch_changepoints(df, min_run) -> dict[Any, list[int]]` mapping TrajectoryID → frame indices where the debounced realtime label changes. Mode-1 errors (mid-track flips with no geometric break) become fragment boundaries the solver can actually fix.

- [ ] **Step 1: Write the failing test**

```python
# tests/identity/test_offline_realtime_splits.py
"""Realtime commitment switches must become fragment boundaries (audit F14:
with PELT off, fragments = whole trajectories, so a mid-track flip was
offline-invisible and majority-vote erased it)."""
import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.identity.offline import _realtime_switch_changepoints


def _df(labels):
    n = len(labels)
    return pd.DataFrame(
        {
            "TrajectoryID": [1] * n,
            "FrameID": range(n),
            C.REALTIME_LABEL: labels,
        }
    )


def test_sustained_switch_is_a_changepoint():
    cps = _realtime_switch_changepoints(_df(["A"] * 100 + ["B"] * 100), min_run=5)
    assert cps == {1: [100]}


def test_blip_shorter_than_min_run_is_ignored():
    cps = _realtime_switch_changepoints(_df(["A"] * 100 + ["B"] * 3 + ["A"] * 100),
                                        min_run=5)
    assert cps == {}


def test_unknown_runs_do_not_split():
    cps = _realtime_switch_changepoints(_df(["A"] * 50 + [""] * 20 + ["A"] * 50),
                                        min_run=5)
    assert cps == {}
```

- [ ] **Step 2: Implement the helper** (place near the other module helpers):

```python
def _realtime_switch_changepoints(
    df: pd.DataFrame, min_run: int
) -> dict[Any, list[int]]:
    """FrameIDs where the debounced realtime label changes, per trajectory.

    Only runs of >= min_run consecutive identical KNOWN labels count; unknown
    /blank rows extend the surrounding run instead of breaking it."""
    col = C.REALTIME_LABEL
    if col not in df.columns:
        return {}
    out: dict[Any, list[int]] = {}
    for traj_id, grp in df.groupby("TrajectoryID", sort=False):
        g = grp.sort_values("FrameID")
        labels = [str(v).strip() for v in g[col].tolist()]
        frames = [int(f) for f in g["FrameID"].tolist()]
        runs: list[tuple[str, int, int]] = []  # (label, start_frame, length)
        for lbl, frame in zip(labels, frames):
            if not lbl or lbl.lower() in ("unknown", "nan", "none"):
                continue
            if runs and runs[-1][0] == lbl:
                runs[-1] = (lbl, runs[-1][1], runs[-1][2] + 1)
            else:
                runs.append((lbl, frame, 1))
        solid = [(lbl, start) for lbl, start, length in runs if length >= min_run]
        cps = [start for (prev, _), (lbl, start) in zip(solid, solid[1:]) if lbl != prev]
        if cps:
            out[traj_id] = cps
    return out
```

- [ ] **Step 3: Wire into `run_fragment_solver`** — immediately before the PELT block:

```python
    if bool(params.get("IDENTITY_SPLIT_ON_REALTIME_SWITCH", True)):
        min_run = max(1, int(params.get("IDENTITY_DISAGREE_MIN_RUN", 5)))
        rt_changepoints = _realtime_switch_changepoints(trajectories_df, min_run)
        if rt_changepoints:
            trajectories_df = split_trajectories_at_changepoints(
                trajectories_df, rt_changepoints, params
            )
            log.info(
                "fragment_solver: split %d trajectories at realtime commitment "
                "switches (audit F14).",
                len(rt_changepoints),
            )
```

FIRST verify `split_trajectories_at_changepoints`'s changepoint format matches `{traj_id: [frame, ...]}` (read its definition and the PELT call site); adapt the helper's return shape to whatever it actually takes. Note: splitting BEFORE PELT means evidence keyed by `OriginalTrajectoryID` still joins correctly — confirm `split_trajectories_at_changepoints` maintains `OriginalTrajectoryID` (the PELT branch relies on the same invariant).

- [ ] **Step 4: End-to-end check** — extend the test file with a solver-level test modeled on `tests/test_fragment_solver.py`'s existing fixtures (read that file; reuse its cache/df builders): a 200-frame trajectory whose realtime label is A then B, with cache evidence agreeing (A-evidence first half, B-evidence second half), must come out with two different `IdentityFinalLabel` values across the two halves — this is the mode-1 repair that was previously impossible. Run new file + `tests/test_fragment_solver.py` + `tests/identity/test_offline_changepoint.py`.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/identity/offline.py tests/identity/test_offline_realtime_splits.py
git commit -m "feat(identity): fragment solver splits at debounced realtime label switches (audit F14)"
```

---

### Task 15: Gap-scaled bridge veto + two-sided velocity z-test (audit fix 11, F15)

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/offline.py` (`_spatial_score_for_fragment` :283-352 and its callers)
- Modify: `src/hydra_suite/core/post/processing.py` (`_compute_velocity_zscore_breaks` :528-531)
- Test: `tests/identity/test_offline_spatial_veto.py`, plus extend the existing z-score test file (`grep -rln "_compute_velocity_zscore_breaks\|MAX_VELOCITY_ZSCORE" tests/` to find it)

**Interfaces:**
- Produces: `_spatial_score_for_fragment(..., diffusion_px: float = 0.0)` — the veto becomes `dist > max_velocity * min(gap, cap) + diffusion_px * sqrt(gap)` (allowance grows with the TRUE gap at diffusion rate instead of freezing at the 30-frame clamp — audit F15: the clamp licensed 1500px teleports at gap=30 while vetoing slower, longer bridges). The Gaussian score still uses `effective_gap = min(gap, cap)`. Callers pass `diffusion_px = 2.0 * body` where body = `params.get("REFERENCE_BODY_SIZE", 20.0) * params.get("RESIZE_FACTOR", 1.0)` — grep `_spatial_score_for_fragment(` for the caller(s) in `_iterative_assign` and thread it from `params`.
- Produces: two-sided z: `z = abs(v - mean) / std` in `_compute_velocity_zscore_breaks`. `MAX_VELOCITY_ZSCORE` default REMAINS 0.0 (feature stays opt-in; flipping the default would change every non-identity pipeline — out of scope per the audit's risk note).

- [ ] **Step 1: Write the failing test**

```python
# tests/identity/test_offline_spatial_veto.py
import pandas as pd

from hydra_suite.core.individual.identity.offline import _spatial_score_for_fragment


def _frag(t0, t1, x0, y0, x1, y1):
    return pd.Series(
        {"StartFrame": t0, "EndFrame": t1, "StartX": x0, "StartY": y0,
         "EndX": x1, "EndY": y1}
    )


def _schedule(end_frame, end_x, end_y):
    return {"A": [{"start_frame": 0, "end_frame": end_frame,
                   "start_X": 0.0, "start_Y": 0.0, "end_X": end_x, "end_Y": end_y}]}


def test_long_gap_slow_bridge_is_no_longer_vetoed():
    # 100-frame gap, 2000px apart: true velocity 20px/f (plausible), but the
    # old 30-frame clamp computed 66px/f > max_velocity=50 -> hard veto.
    frag = _frag(200, 250, 2000.0, 0.0, 2100.0, 0.0)
    score, has_n = _spatial_score_for_fragment(
        frag, "A", _schedule(100, 0.0, 0.0), max_velocity=50.0,
        diffusion_px=40.0,
    )
    assert has_n and score > 0.0


def test_fast_teleport_within_cap_is_still_vetoed():
    # 2-frame gap, 2000px: 1000px/f — must remain a hard veto.
    frag = _frag(102, 150, 2000.0, 0.0, 2100.0, 0.0)
    score, has_n = _spatial_score_for_fragment(
        frag, "A", _schedule(100, 0.0, 0.0), max_velocity=50.0,
        diffusion_px=40.0,
    )
    assert has_n and score == 0.0
```

- [ ] **Step 2: Implement** — in `_spatial_score_for_fragment`, add keyword `diffusion_px: float = 0.0` and change each of the two neighbor blocks from:

```python
        gap = max(1, t0 - prior["end_frame"])
        effective_gap = min(gap, cap)
        dist = math.hypot(...)
        velocity = dist / effective_gap
        if velocity > max_velocity:
            return 0.0, True
```

to:

```python
        gap = max(1, t0 - prior["end_frame"])
        effective_gap = min(gap, cap)
        dist = math.hypot(x0 - prior["end_X"], y0 - prior["end_Y"])
        allowance = max_velocity * effective_gap + diffusion_px * math.sqrt(gap)
        if dist > allowance:
            return 0.0, True  # physically implausible — hard veto
        velocity = dist / effective_gap
        velocity = min(velocity, max_velocity)  # score saturates at 1 sigma
```

(mirror in the `following` block; update the function docstring's veto paragraph). Thread `diffusion_px` from the caller(s) in `_iterative_assign` (grep; compute from params as in Interfaces). In `processing.py`'s `_compute_velocity_zscore_breaks`, change the z computation to `z = np.abs(v - mean) / std` (read the surrounding 20 lines first; keep everything else, including the `MAX_VELOCITY_ZSCORE <= 0` disable).

- [ ] **Step 3: Run** — new test file, the located z-score tests (a one-sided expectation may need updating to |z|), `tests/test_fragment_solver.py`, and the retention benchmark.

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/core/individual/identity/offline.py src/hydra_suite/core/post/processing.py tests/
git commit -m "fix(identity): diffusion-scaled bridge veto + two-sided velocity z (audit F15)"
```

---

### Task 16: Cumulative pipeline gate on real fixtures (MPS + CUDA)

**Files:**
- Create: `docs/superpowers/plans/2026-08-13-identity-retention-fixes-results.md` (results record)

**Interfaces:**
- Consumes: `tools/equivalence/run_matrix.sh` (existing), `tools/identity_bench/identity_metrics.py` (Task 2).
- Acceptance (identity clips `emi_obb_identity`, `ant_cnn_identity`, comparing branch vs branch-point baseline): `realtime_flips_per_1k` strictly lower or equal on both clips with at least one strictly lower; `teleports_per_1k` not higher; `unknown_fraction` increase ≤ 0.10 absolute; every produced CSV has `wc -l` > 1. Non-identity clips `fly_obb`, `worm_bgsub`: EQUIVALENCE at the determinism floor (byte-identical — they execute none of the changed code). DETERMINISM must be clean for all clips (new_a vs new_b identical — the fixes must not introduce nondeterminism).

- [ ] **Step 1: Prepare** — from the `.worktrees/identity-fixes` worktree root:

```bash
# kill only dead/stale sleap/hydra processes (inspect before killing; never touch others)
ps aux | grep -E "sleap|hydra" | grep -v grep
conda activate hydra-mps                      # MUST be active (empty-CSV gotcha)
bash tools/equivalence/fixtures/fetch_fixtures.sh   # no-op if fixtures present
BASE_SHA=$(git merge-base HEAD main)
git worktree add --detach .worktrees/identity-baseline "$BASE_SHA"
```

- [ ] **Step 2: Run the matrix (MPS)**

```bash
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/identity-baseline/src WT_SRC=$PWD/src \
  OUT=/tmp/identity_bench RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh emi_obb_identity ant_cnn_identity fly_obb worm_bgsub
find /tmp/identity_bench -name "*_tracking_final.csv" -exec wc -l {} \;   # all > 1
```

- [ ] **Step 3: Score** — for each identity clip, run the metrics CLI on the baseline (`legacy`/MAIN run) and current (`new_a`) final + forward CSVs:

```bash
for f in $(find /tmp/identity_bench -name "*_tracking_final.csv"); do
  echo "== $f"; PYTHONPATH=$PWD/src:$PWD conda run -n hydra-mps \
    python tools/identity_bench/identity_metrics.py "$f"
done
```

Record every JSON blob in the results file with a baseline-vs-current table per clip and a PASS/FAIL against each acceptance criterion. Also record the harness's printed DETERMINISM/EQUIVALENCE/PERFORMANCE lines: fly_obb + worm_bgsub must print EQUIVALENT at the noise floor; identity clips are expected to print differences (note them as expected).

- [ ] **Step 4: Ablation spot-check (attribution)** — rerun ONE identity clip (`emi_obb_identity`) with the new knobs forced to their legacy values via the fixture config (copy the clip's config, set `identity_per_frame_evidence_cap: 0`, `identity_prob_floor: 0`, `identity_evidence_tau: 1`, `identity_commit_revision_min_frames: 1`, `identity_rejoin_confirm_frames: 1`, point the runner at the copied config — see `tools/equivalence/README.md` for how the runner resolves configs). Metrics for this run should land near the baseline's, confirming the improvement is attributable to the knobs, not incidental drift. Record in the results file. If the harness does not support per-clip config overrides, note that and substitute the scenario-suite ablation (each fix's target test run with legacy knob values must fail) as the attribution evidence.

- [ ] **Step 5: CUDA confirmation (mehek)** — per repo convention:

```bash
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite && git fetch origin && git checkout feat/identity-retention-fixes  # push the branch first
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
git worktree add --detach .worktrees/identity-baseline $(git merge-base HEAD origin/main)
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/identity-baseline/src WT_SRC=$PWD/src \
  OUT=/tmp/identity_bench RUNTIME=cuda nohup bash tools/equivalence/run_matrix.sh \
  emi_obb_identity ant_cnn_identity fly_obb worm_bgsub > /tmp/identity_bench.log 2>&1 &
```

Same scoring + acceptance as Steps 3-4 (metrics deltas must point the same direction; fly/worm EQUIVALENT). Record in the results file. Clean up both baseline worktrees afterwards (`git worktree remove --force .worktrees/identity-baseline && git worktree prune`, on both machines).

- [ ] **Step 6: Final hygiene + commit**

```bash
make format && make lint          # from the worktree; fix anything introduced
PYTHONPATH=$PWD/src:$PWD conda run -n hydra-mps python -m pytest \
  tests/identity/ tests/test_fragment_solver.py -v      # full identity battery green
git add docs/superpowers/plans/2026-08-13-identity-retention-fixes-results.md
git commit -m "docs(identity): retention-fix pipeline gate results (MPS + CUDA)"
```

Do NOT merge to main in this task — merging is a separate user decision after reviewing the results file.

---

## Self-review notes

- **Spec coverage:** audit §6 items 1-13 all mapped (see Fix → Task table). Item 7's "full de-double-counting" half is intentionally deferred (audit marks it "needs care"); Task 11 ships the consistency half the audit rates low-risk, plus the weight-default alignment from open question 6.
- **Known judgment calls baked in:** lock bias = log-odds bonus (not mixture — a mixture with s>0.5 pins the argmax unconditionally); τ/M defaults tunable per Task 7 Step 5's explicit rule; `MAX_VELOCITY_ZSCORE` default stays 0.0; AprilTag unknown floor unchanged.
- **Execution risk concentrations:** Task 11 (worker rewiring — read the fusion block before touching it) and Task 14 (changepoint-format contract — verify before wiring). Both tasks carry explicit "read first, verify contract" steps.
