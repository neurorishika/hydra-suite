# Adversarial audit — the identity-retention plan (2026-09-06)

**Verdict: do not execute the plan as written.**

Four independent Fable agents in refute-mode audited
`docs/superpowers/plans/2026-08-13-identity-retention-fixes.md` and its spec
`docs/superpowers/specs/2026-08-13-identity-audit.md` against `main` @79db3106.
Every verdict below is backed by a probe that drove the real
`OnlineIdentityDecoder` / `TrackAssigner` / `core.post.processing` in-process.
Probe scripts: `/tmp/audit-{a,b,c,d}/`.

## 1. The audit's diagnosis is mostly right

`online.py` has not changed since 2026-08-08 (4851a765) — before the audit was
written. Every F3/F4/F5/F8 line it cites is byte-identical today; the
post-audit identity work (repair @f2d4ca36, final-consistency @5a6bb2f8) touched
the fragment solver and the worker, not the online decoder.

Reproduced on today's code:

- **F1 uncapped fusion.** Cold slot at 0.868 confidence after 2 frames of
  0.6-confidence evidence; commits on frame 5 (`commit_min_hits=5` binds).
  A committed **and slot-locked** identity on an isolated animal flips A→B after
  exactly 5 frames of wrong 0.6 evidence, and back 5 frames later.
  `per_frame_cap`/`prob_floor` are genuinely dead: the only grep hits are the
  definitions; `RobustnessConfig` still reads "Reserved (Phase 3)".
- **F3 slot-lock sign.** Confirmed and *understated*. For a single-entry shift,
  `p' = p·e^b/(p·e^b + 1−p)` is monotone in `b`, so `log(0.9) < 0` strictly
  lowers the locked entry even after renormalisation. Compounding is the damage:
  a locked slot fed missing evidence decays below 0.6 in **14 frames with the
  bias vs 31 without**. The lock halves the memory horizon of what it protects.
- **F4 margin tautology.** Incumbent and challenger confidences come from the
  same normalised simplex (`assignment.confidence == p[assigned] ==
  p[committed] == 0.906360`). Vacuous iff `2·commit_threshold − 1 ≥ margin`
  ⇔ threshold ≥ 0.75; shipped threshold is 0.85, so blocked = 0 always.
- **F12/F13/F14/F15 premises** all hold, with line anchors ~200 lines drifted.
- **Motion enters identity nowhere** — literally true: `update_frame(frame_idx,
  visible_slots, slot_evidences)` takes no geometry.

## 2. Four headline claims are wrong

| Claim | Reality |
|---|---|
| F1 "confidence inflates linearly with **no ceiling**" | **Refuted.** The sticky transition leak (`transition_epsilon/(C−1)`) is a hard ceiling at log-odds 6.53, p≈0.995, flat for 200 frames. Practically irrelevant (0.995 ≫ 0.85 commit), but the plan reasons from an unbounded model. |
| F2 "p(unknown) → 1e-8 after one frame" | **Stale.** The repair's `unknown_prior=0.05` (`substrate.py:370-376`, 84e8270f) puts it at 4.9e-4. Mechanism survives (unknown is still unreachable by accumulation: 0.0011 after 40 ambiguous frames) but the magnitude and the cited path are both wrong — Appendix A probed `catalog.cnn_log_prior`, which feeds only an off-by-default association addon, **not** the decoder. |
| F7 "6000 px teleport after 100 lost frames" | **Refuted as stated.** The budget is linear but the worker decays committed-lost beliefs each frame; at 100 frames p(A)=0.24, score 0.236 < 0.5 — no rejoin. Real reach: ~1140 px @0.55 conf, ~2940 px @0.95, all within ~1.5 s of loss. Task 9's rationale targets a regime that cannot fire. |
| F9 "recoverable only via rejoin" | **Refuted.** `hungarian.py:1199-1203` demotes an unmatched committed-lost slot to the proximity path; it is reused whenever the animal reappears within `MAX_DISTANCE_THRESHOLD`. Deadlock needs reappearance beyond MAX_DIST from *every* committed-lost slot. |

F8's mechanism is also mis-stated: the swap counter survives missing evidence
(a swap fired straight through a MISSING frame) and is dropped only by a
**visibility** gap. And while the challenger label's owner is a live slot, the
isolated flip is *pinned*, not flapping.

## 3. A defect with no task — F5 corrupts the CSV

For a committed slot whose assignment disagrees, `worker.py:1961-1964`
overwrites p(committed) with the *assignment's* confidence. Measured two-slot
output:

```
slot 0: belief p(A)=0.001 p(B)=0.999 committed=A -> CSV  A, 0.999
slot 1: belief p(A)=0.001 p(B)=0.999 committed=B -> CSV  B, 0.000
```

`IdentityRealtimeConfidence` is not the confidence of the label in its own row.
Which slot receives the `0.000` depends on a Hungarian tie-break and alternates
frame to frame. **F5 appears nowhere in the plan's fix→task map.** In the
two-slot regime this — not a label flip — is the symptom that reaches the user.

## 4. The oracle does not work

Task 1's harness and Task 2's metrics CLI are the plan's entire evidence base.
Both are unsound:

- **The harness models the wrong evidence path.** It builds evidence via
  `catalog.cnn_log_prior` (unknown pinned at 1e-6). The decoder is fed the
  sidecar through `map_cnn_to_catalog(unknown_prior=0.05)` + `remap_phase_log_probs`.
  Every retention target in Tasks 5/7/8 is measured on a vector the decoder
  never receives.
- **The harness is single-slot**, so it structurally cannot observe the
  other-slot commit block (`online.py:530-541`), the F5 row corruption, or the
  visibility-gap counter reset — i.e. the entire multi-animal regime.
- **A fix that *hurts* retention passes every gate.** "Commit less" and "never
  rejoin" satisfy all of Task 16's criteria. Task 9's threshold clamp is itself
  such a fix (§5).
- Tautological targets, individually verified: Task 3 (`biased >= raw` holds if
  you simply delete the lock; the real fix changes **zero** scenario outcomes),
  Task 7 (the tempering test passes for τ ∈ {2,3,5,10}: 0.003–0.059 against a
  0.15 threshold), Task 11 (asserts `out[0] == cal`, its own input, and imports
  a helper that does not exist), Task 13 (`long > short` is true of any monotone
  statistic), Task 10 (tests the counter, not reachability).
- **`final_label_switches` ≡ 0 by construction** of per-fragment assignment, and
  final-CSV teleports are diluted by the dense interior fill shipped in ec82ea5c.

## 5. Two tasks are actively harmful

- **Task 9 makes rejoin unreachable on the only live identity fixture.**
  `ant_cnn_identity` has `confidence: 0.9`; the clamp is `max(threshold,
  display)`. Best-case sidecar compatibility is 0.9405 < 0.95, and
  `hungarian.py` tests strictly `>`. After Tasks 9+11 no rejoin can ever fire.
- **Task 9's confirmation window is undone on frame 1.** While the window is
  pending the identity branch does not fire, so the assigner's own fallback
  (`hungarian.py:1199-1203`) demotes the slot to a proximity respawn: new
  trajectory ID, hard KF reset, `clear_slot`, streak dict orphaned. Task 9
  converts would-be rejoins into exactly the fragmentation rejoin prevents.
- **Task 10 gates the already-gated path and misses the ungated one.** Its
  spatial gate wraps the single `clear_slot` site, reached only for proximity
  respawns already bounded by MAX_DIST. The bootstrap path never calls
  `clear_slot` — that is where a stale committed belief carries at full strength
  with no gap cap (measured: a 0.999-A belief reused 497 frames later reports
  `label=A p=0.96` on its first frame).

## 6. "Every knob has a legacy value" is false — three ways

| Knob | Claim | Reality |
|---|---|---|
| `IDENTITY_PER_FRAME_EVIDENCE_CAP=0` | off | **Dangerous.** At the substrate `per_frame_cap=0` → `np.isfinite(0)` → `clip(·,−0,0)` = total evidence annihilation. "Off" exists only via Task 5's `≤0 → inf` translation; any call site passing the key straight through silently zeroes all evidence. The substrate's real no-op is `inf`. |
| `IDENTITY_EVIDENCE_TAU=1` | no-op | True online (bitwise). **False offline** — τ=1 gives a plain SUM where legacy is a MEAN; `_combined_support` saturates to 1.000/0.000 at n=100, breaking `FRAGMENT_MIN_SUPPORT`. |
| `IDENTITY_REJOIN_CONFIRM_FRAMES=1` | reproduces old behaviour | **False.** Task 9's √t budget and `max(threshold, display)` clamp are unconditional: budget 600 → 190 px, threshold 0.5 → 0.6 even at confirm=1. |

`PROB_FLOOR=0`, `COMMIT_REVISION_MIN_FRAMES=1`, `SPLIT_ON_REALTIME_SWITCH=False`
and `MAX_VELOCITY_ZSCORE=0.0` are genuine no-ops. But Tasks 8, 9, 10, 11, 12,
13 and 15 all carry **knob-less** changes (incl. `association_weight` 1.0→0.3,
a 3.3× coupling change for every user), so Task 16 Step 4's "ablate to baseline"
cannot land near baseline.

## 7. The gate is one clip, and it is the wrong one

**`emi_obb_identity` executes no identity code.** Its config is
`identity_method: "none_disabled"` with `cnn_classifiers: []`, so
`resolve_catalog_spec` returns `entries=()`, no decoder is constructed
(`worker.py:1895-1912`) and `run_fragment_solver` exits at `if not
known_labels`. Despite the name, it must stay byte-identical.

That leaves **`ant_cnn_identity` as the only live identity fixture** for a
16-task gate. Its classifier is the unstamped April colortag model
(`fit_policy: ABSENT` → resolves to `squash`) — correct post-repair, but it
means there is still zero coverage of a *stamped* model, which is the repair's
own open follow-up. `fly_obb`/`worm_bgsub` byte-identity proves only that
`_identity_online_decoder is None` there.

## 7b. Seven decoder knobs are unreachable and unrecorded

`online.py` reads 14 `IDENTITY_*` keys; `engine_params.py` emits 7. These are
read but never emitted, so they take a hardcoded default that nothing in the run
record distinguishes from a deliberate choice — and neither `advanced_config`
nor a top-level config override reaches `params` (both verified `False`), so a
user cannot set them even knowing they exist:

```
IDENTITY_COMMIT_MIN_HITS
IDENTITY_SLOT_LOCK_MIN_FRAMES / _STRENGTH / _OVERRIDE_MARGIN
IDENTITY_RESPAWN_PRIOR_STRENGTH / _DECAY / _MAX_GAP
```

Two are load-bearing on the results above. `IDENTITY_COMMIT_MIN_HITS=5` is the
actual binding gate on commit latency — tempering evidence 5x does not move
cold-commit off 5 frames, because the hit count and not the posterior decides.
`IDENTITY_SLOT_LOCK_STRENGTH=0.9` enters as `log(0.9) < 0`, the F3 sign defect.

Task 4 is the plan's answer to this, but its premise is half wrong: exactly
these 7 are hardcoded-and-unemitted, while 6 of the keys it lists
(`..._CAP`, `..._PROB_FLOOR`, `..._EVIDENCE_TAU`, `..._COMMIT_REVISION_MIN_FRAMES`,
`..._REJOIN_CONFIRM_FRAMES`, `..._SPLIT_ON_REALTIME_SWITCH`) are read by nothing
— they are new knobs, not existing ones. Task 4's line anchors are also stale
(`engine_params.py:1411-1443`, not `:1113-1146`).

A precedent worth copying once this is scoped: `core/inference/geometry_drift.py`
(6b866d78) logs a value **and its source** (explicit / profile / derived /
contract default) at every build, because a silently-taken default invalidated a
SAM3 model comparison. Identity needs emit-then-log — there is currently no
emitted value for which to record a source.

## 8. Task conflicts

- **8 ↔ 11** — Task 8 edits `_factor_log_prob`, whose result is overwritten 100
  lines later by `map_cnn_to_catalog`'s `probs[0] = unknown_prior`; its other
  edit targets `cnn_log_prior`, the association path Task 11 deletes. Dead on
  arrival on the live path.
- **8 depends on 5+7, unstated.** Task 8 alone: p(unknown) 0.026, still commits.
  Stacked with 5+7: 0.117, does not commit. Its Step-3 STOP misfires on reorder.
- **5 ↔ 7** — at τ=5, cap=1.0 never binds on live evidence for confidence ≲0.97.
  On the harness's evidence it binds only through the 1e-6 unknown entry.
- **3 ↔ 6** — fixing the lock sign makes the revision gate Task 6 tunes largely
  unreachable (latency 3→4 and 7→9 frames from Task 3 alone; 11 and 16 with both).
  Neither task mentions the coupling.
- **13 ↔ d6a1c4d5/81d29006**, **14 ↔ 8252cc4d/b246724b/77ef4988** — the fixes
  contradict shipped behaviour (Task 14's wiring bypasses `breaker_tripped` and
  never sets `did_split`).
- **7 ↔ AprilTags** — tempering hits AprilTag evidence too (~9 → ~1.8 nats),
  and no fixture covers AprilTags at all.

## 9. Does it meet its goal?

No. With Tasks 3+5+6+7 stacked at plan defaults, a 40-frame 0.9-confidence
wrong streak **still produces the A→B→A double flip**; cold commit stays 5–6
frames. The plan moves the flap boundary from ~5 frames to somewhere in 13–40
and leaves the commit/lock/revision/swap state machine — which spec §7 names as
the root flaw — untouched. Several of the plan's own invariants *require* the
flap to survive.

## 10. What shipped from this audit (2026-09-06)

F5, F3 and F4-as-a-warning shipped on `fix/identity-honesty-slice`
(705a5999, 55b85729, 66fc7e77). Recorded here because §11 below is written
against the pre-fix code.

- **F5 fixed.** `out_conf = probs[out_idx]`: the reported confidence now
  belongs to the reported label by construction.
- **F3 fixed, with a caveat.** The bias is now `log1p(strength)` (raises the
  locked entry) and is applied to a *copy* inside the assignment solve, so it
  can no longer compound. **It does not, however, protect anything at shipped
  defaults**: the bias feeds only the Hungarian solve (gated by
  `display_threshold` 0.6) while revision is decided on the raw posterior
  against `commit_threshold` 0.85, so a challenger at 0.85 is assigned
  regardless. Locked and unlocked decoders revise on the identical frame. The
  fix removes the old harm; it adds no protection. `-log1p(-strength)`
  (+2.303 nats) *does* block revisions and fails exactly one existing test
  whose config locks after a single frame — re-evaluate both once an oracle
  exists.
- **F4 not fixed, announced.** Retuning needs an oracle. The decoder now warns
  once per `(margin, threshold)` pair when the gate cannot bind.
- **3 of the 7 unreachable knobs** (§7b) are now emitted *and* wired through
  `from_engine_config`, defaults unchanged.

**Gate (MPS, baseline `main` @9ae13148):** `worm_bgsub` and `fly_obb` are
byte-identical including every keyed column. `ant_cnn_identity` and
`ant_cnn_identity_relink` have identical positions/θ/row counts/unmatched,
with exactly 5 columns differing — `IdentityRealtimeID`, `...Label`,
`...Confidence`, `...Margin`, `...SlotLock`. So the change is confined to the
identity columns and does **not** move trajectories, despite these fixtures
enabling the online decoder and its association-cost addon.

Two gate caveats worth carrying forward: **`ant_cnn_identity_marked` never
ran** — its config exists but `ant_cnn_identity_marked.mp4` does not, and the
harness still printed "all clips produced comparable output"; and `fly_obb`
reported 1.26x perf against a 1.25x tolerance while an unrelated `trackerkit`
GUI held the machine (other clips: 0.43x/0.85x/1.11x).

## 11. Recommended reshape

1. **Fix F5 first.** One-line class of bug, corrupts the shipped CSV, has no
   task, needs no new machinery.
2. **Fix F3's sign and F4's threshold condition** (`margin > 2·threshold − 1`)
   as two small independent commits with bitwise-diff oracles, not xfail flips.
3. **Rebuild the oracle before any behavioural fix.** It must (a) feed evidence
   through `map_cnn_to_catalog` + `remap_phase_log_probs`, (b) be multi-slot,
   (c) score against held-out ground truth, and (d) demonstrably fail for
   "commit less" and "never rejoin".
4. **Add a second live identity fixture** — `emi_obb_identity` is misnamed and
   inert; one clip cannot gate 16 tasks. Pair with the repair's open follow-up
   (a stamped classifier fixture).
5. **Drop or restate Tasks 8, 9, 10, 13, 14** — each targets a path that the
   shipped work has moved, or that was never the damaging one.
6. **Re-derive Tasks 5/7 targets on live evidence.** Every number in them was
   measured on a vector the decoder does not receive.
