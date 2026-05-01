"""Online identity decoder with uniqueness enforcement.

Identity Phases 1 & 2: runs after geometric assignment and before final
identity publication for each tracked frame.

Algorithm per-frame
-------------------
1. Predict each active slot's belief forward with a sticky Markov transition.
2. Fuse all matched evidence items (AprilTag + CNN) for that slot in log-space.
3. Optionally apply a soft slot-lock bias to strongly committed slots.
4. Solve a visible-slot → known-identity assignment with dummy unassigned
   columns, enforcing the partial-injective (uniqueness) constraint.
5. Update commitment state and soft slot-lock counters.
6. Emit one ``IdentityAssignment`` per visible slot.

Configuration keys (all read from params dict, with safe defaults)
------------------------------------------------------------------
IDENTITY_TRANSITION_EPSILON          float  transition spread (default 0.02)
IDENTITY_UNKNOWN_PRIOR               float  prior mass on unknown (default 0.05)
IDENTITY_COMMIT_THRESHOLD            float  confidence to commit (default 0.85)
IDENTITY_COMMIT_MIN_HITS             int    evidence hits to commit (default 5)
IDENTITY_DISPLAY_THRESHOLD           float  min confidence to display (default 0.6)
IDENTITY_SLOT_LOCK_MIN_FRAMES        int    frames before slot lock (default 30)
IDENTITY_SLOT_LOCK_STRENGTH          float  soft-lock bias weight (default 0.9)
IDENTITY_SLOT_LOCK_OVERRIDE_MARGIN   float  margin to override lock (default 0.5)
IDENTITY_SWAP_ENABLED                bool   enable swap-correction (default True)
IDENTITY_SWAP_MIN_FRAMES             int    sustained mutual-mismatch frames (default 8)
IDENTITY_SWAP_CONF_MARGIN            float  prob margin to count as mismatch (default 0.2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from hydra_suite.core.identity.catalog import IdentityCatalog
from hydra_suite.core.identity.evidence import IdentityEvidence

log = logging.getLogger(__name__)


@dataclass
class TrackIdentityBelief:
    """Running probabilistic belief over the identity catalog for one slot.

    ``log_posterior`` is shape ``(catalog_size,)`` float64 in log-scale.
    Index 0 is always the unknown/unobserved state.

    Parameters
    ----------
    slot_index:
        Track slot index (matches ``KalmanFilterManager`` slot numbering).
    log_posterior:
        Current log-posterior over the full catalog.
    hit_count:
        Total number of frames where at least one evidence item was fused.
    stable_count:
        Consecutive frames agreeing with the current committed identity.
        Resets to 0 on any commitment change.
    committed:
        True once the belief has crossed both the confidence and hit thresholds.
    committed_label:
        The identity label at commitment time.
    committed_index:
        Catalog index of ``committed_label``.
    slot_lock_label:
        Identity soft-locked to this slot (may be None before lock triggers).
    slot_lock_strength:
        Fraction of probability mass the lock diverts toward the locked label.
    slot_lock_frame:
        Absolute frame index when the lock was first applied.
    """

    slot_index: int
    log_posterior: np.ndarray
    hit_count: int = 0
    stable_count: int = 0
    committed: bool = False
    committed_label: Optional[str] = None
    committed_index: int = 0
    slot_lock_label: Optional[str] = None
    slot_lock_strength: float = 0.0
    slot_lock_frame: int = 0
    last_frame_idx: int = -1
    last_evidence_sources: tuple[str, ...] = field(default_factory=tuple)
    last_conflict_flag: bool = False


@dataclass(frozen=True)
class RespawnPrior:
    """Decayed prior carried across slot reuse after temporary loss."""

    slot_index: int
    log_posterior: np.ndarray
    committed_label: Optional[str] = None
    committed_index: int = 0
    last_frame_idx: int = -1


@dataclass(frozen=True)
class IdentityAssignment:
    """Per-slot identity assignment result emitted for one frame.

    Emitted by ``OnlineIdentityDecoder.update_frame()``.

    Parameters
    ----------
    slot_index:
        Track slot this assignment covers.
    label:
        Assigned identity label, or ``None`` when the slot is unassigned or
        below the display threshold.
    catalog_index:
        Catalog index of *label* (0 = unknown when unassigned).
    confidence:
        Posterior probability of the assigned label.
    entropy:
        Shannon entropy of the full posterior (nats).
    margin:
        Difference between the top-1 and top-2 known-identity posteriors.
        Higher margin = more decisive.
    committed:
        True when the slot has reached the commitment threshold.
    """

    slot_index: int
    label: Optional[str]
    catalog_index: int
    confidence: float
    entropy: float
    margin: float
    committed: bool


class OnlineIdentityDecoder:
    """Online Bayesian identity decoder with uniqueness enforcement.

    One instance per tracking run.  ``update_frame()`` is called once per
    frame from the tracking loop; ``clear_slot()`` is called when a lost slot
    is respawned.

    The decoder does not modify the cost matrix or the geometric assigner;
    identity uniqueness is enforced inside the decoder's own visible-slot
    assignment step and is communicated back to the worker via the returned
    ``IdentityAssignment`` objects.
    """

    def __init__(self, catalog: IdentityCatalog, params: dict[str, Any]) -> None:
        self._catalog = catalog
        self._params = params
        self._beliefs: dict[int, TrackIdentityBelief] = {}
        self._respawn_priors: dict[int, RespawnPrior] = {}

        # -- Configuration with safe defaults --
        self._transition_epsilon: float = float(
            params.get("IDENTITY_TRANSITION_EPSILON", 0.02)
        )
        self._unknown_prior: float = float(params.get("IDENTITY_UNKNOWN_PRIOR", 0.05))
        self._commit_threshold: float = float(
            params.get("IDENTITY_COMMIT_THRESHOLD", 0.85)
        )
        self._commit_min_hits: int = int(params.get("IDENTITY_COMMIT_MIN_HITS", 5))
        self._display_threshold: float = float(
            params.get("IDENTITY_DISPLAY_THRESHOLD", 0.6)
        )
        self._slot_lock_min_frames: int = int(
            params.get("IDENTITY_SLOT_LOCK_MIN_FRAMES", 30)
        )
        self._slot_lock_strength: float = float(
            params.get("IDENTITY_SLOT_LOCK_STRENGTH", 0.9)
        )
        self._slot_lock_override_margin: float = float(
            params.get("IDENTITY_SLOT_LOCK_OVERRIDE_MARGIN", 0.5)
        )
        self._respawn_prior_strength: float = float(
            params.get("IDENTITY_RESPAWN_PRIOR_STRENGTH", 0.75)
        )
        self._respawn_prior_decay: float = float(
            params.get("IDENTITY_RESPAWN_PRIOR_DECAY", 0.97)
        )
        self._respawn_prior_max_gap: int = int(
            params.get("IDENTITY_RESPAWN_PRIOR_MAX_GAP", 120)
        )
        self._swap_enabled: bool = bool(params.get("IDENTITY_SWAP_ENABLED", True))
        self._swap_min_frames: int = int(params.get("IDENTITY_SWAP_MIN_FRAMES", 8))
        self._swap_conf_margin: float = float(
            params.get("IDENTITY_SWAP_CONF_MARGIN", 0.2)
        )

        # Sustained-mutual-mismatch counter, keyed by sorted slot pair.
        self._swap_evidence: dict[tuple[int, int], int] = {}

        # Build sticky transition matrix in log-space once
        self._log_transition: np.ndarray = self._build_log_transition(catalog.size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_log_transition(self, C: int) -> np.ndarray:
        """Build log-space sticky transition matrix of shape (C, C)."""
        eps = self._transition_epsilon
        T = np.full((C, C), eps / max(C - 1, 1), dtype=np.float64)
        np.fill_diagonal(T, 1.0 - eps)
        return np.log(np.clip(T, 1e-300, None))

    def _initial_log_posterior(
        self,
        blocked_labels: Optional[set[str]] = None,
    ) -> np.ndarray:
        """Weak log-prior with optional downweighting of claimed identities."""
        blocked = {str(label) for label in (blocked_labels or set()) if label}
        if not blocked:
            return self._catalog.known_uniform_log_prior(
                unknown_weight=self._unknown_prior
            )

        priors = np.full(self._catalog.size, 1e-300, dtype=np.float64)
        priors[0] = self._unknown_prior
        available_indices = [
            idx
            for idx in self._catalog.known_indices()
            if self._catalog.label_of(idx) not in blocked
        ]
        if not available_indices:
            return self._catalog.known_uniform_log_prior(
                unknown_weight=self._unknown_prior
            )

        known_mass = max(1e-6, 1.0 - self._unknown_prior)
        per_label = known_mass / float(len(available_indices))
        for idx in available_indices:
            priors[idx] = per_label
        priors /= np.clip(priors.sum(), 1e-300, None)
        return np.log(np.clip(priors, 1e-300, None))

    def _renormalize_log_probs(self, log_probs: np.ndarray) -> np.ndarray:
        out = np.asarray(log_probs, dtype=np.float64).copy()
        out -= np.logaddexp.reduce(out)
        return out

    def _mask_blocked_labels(
        self,
        log_probs: np.ndarray,
        blocked_labels: Optional[set[str]] = None,
    ) -> np.ndarray:
        blocked = {str(label) for label in (blocked_labels or set()) if label}
        if not blocked:
            return self._renormalize_log_probs(log_probs)
        masked = np.asarray(log_probs, dtype=np.float64).copy()
        available = False
        for label in blocked:
            if not self._catalog.contains(label):
                continue
            idx = self._catalog.index_of(label)
            masked[idx] = np.log(1e-300)
        for idx in self._catalog.known_indices():
            if self._catalog.label_of(idx) in blocked:
                continue
            available = True
            break
        if not available:
            return self._initial_log_posterior(blocked_labels=None)
        return self._renormalize_log_probs(masked)

    def _respawn_seed_log_posterior(
        self,
        slot_index: int,
        frame_idx: Optional[int],
        blocked_labels: Optional[set[str]] = None,
    ) -> np.ndarray:
        base_log = self._initial_log_posterior(blocked_labels)
        carried = self._respawn_priors.pop(slot_index, None)
        if carried is None or frame_idx is None or self._respawn_prior_strength <= 0.0:
            return base_log

        gap = max(0, abs(int(frame_idx) - int(carried.last_frame_idx)) - 1)
        if gap > self._respawn_prior_max_gap:
            return base_log

        carry_strength = float(
            np.clip(
                self._respawn_prior_strength * (self._respawn_prior_decay**gap),
                0.0,
                0.999,
            )
        )
        if carry_strength <= 0.0:
            return base_log

        base_probs = self._posterior_from_log(base_log)
        carried_log = self._mask_blocked_labels(carried.log_posterior, blocked_labels)
        carried_probs = self._posterior_from_log(carried_log)
        mixed = ((1.0 - carry_strength) * base_probs) + (carry_strength * carried_probs)
        mixed /= np.clip(mixed.sum(), 1e-300, None)
        return np.log(np.clip(mixed, 1e-300, None))

    @staticmethod
    def _posterior_from_log(log_probs: np.ndarray) -> np.ndarray:
        shifted = np.asarray(log_probs, dtype=np.float64) - np.max(log_probs)
        probs = np.exp(shifted)
        return probs / np.clip(probs.sum(), 1e-300, None)

    def _get_or_create_belief(
        self,
        slot_index: int,
        blocked_labels: Optional[set[str]] = None,
        frame_idx: Optional[int] = None,
    ) -> TrackIdentityBelief:
        if slot_index not in self._beliefs:
            self._beliefs[slot_index] = TrackIdentityBelief(
                slot_index=slot_index,
                log_posterior=self._respawn_seed_log_posterior(
                    slot_index,
                    frame_idx,
                    blocked_labels,
                ),
                last_frame_idx=int(frame_idx) if frame_idx is not None else -1,
            )
        return self._beliefs[slot_index]

    def _predict_belief(self, belief: TrackIdentityBelief) -> None:
        """Apply sticky transition: prior = T^T · posterior (log-space)."""
        log_post = belief.log_posterior
        new_log = np.empty_like(log_post)
        for j in range(len(log_post)):
            new_log[j] = np.logaddexp.reduce(log_post + self._log_transition[:, j])
        belief.log_posterior = new_log

    def _fuse_evidence(
        self,
        belief: TrackIdentityBelief,
        evidences: list[IdentityEvidence],
    ) -> None:
        """Fuse a list of evidence items via log-space addition (Bayesian update)."""
        for ev in evidences:
            if len(ev.log_probs) != self._catalog.size:
                log.warning(
                    "Slot %d: evidence catalog size %d != decoder catalog size %d; skipping",
                    belief.slot_index,
                    len(ev.log_probs),
                    self._catalog.size,
                )
                continue
            belief.log_posterior = belief.log_posterior + ev.log_probs
            # Renormalise to prevent float underflow over long runs
            belief.log_posterior -= np.logaddexp.reduce(belief.log_posterior)
            belief.hit_count += 1

    def _apply_slot_lock_bias(self, belief: TrackIdentityBelief) -> None:
        """Apply soft slot-lock bias toward the locked identity (Phase 2)."""
        if not belief.slot_lock_label:
            return
        try:
            lock_idx = self._catalog.index_of(belief.slot_lock_label)
        except KeyError:
            return
        # Boost the locked label; renormalise
        log_bias = np.log(max(belief.slot_lock_strength, 1e-6))
        belief.log_posterior[lock_idx] += log_bias
        belief.log_posterior -= np.logaddexp.reduce(belief.log_posterior)

    def _posterior_probs(self, belief: TrackIdentityBelief) -> np.ndarray:
        """Return normalised probability vector for a belief."""
        lp = belief.log_posterior - belief.log_posterior.max()
        p = np.exp(lp)
        return p / p.sum()

    @staticmethod
    def _entropy(probs: np.ndarray) -> float:
        safe = np.clip(probs, 1e-300, None)
        return float(-np.sum(probs * np.log(safe)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_frame(
        self,
        frame_idx: int,
        visible_slots: list[int],
        slot_evidences: dict[int, list[IdentityEvidence]],
    ) -> list[IdentityAssignment]:
        """Process one frame: predict → fuse → assign → commit.

        Parameters
        ----------
        frame_idx:
            Current absolute frame index.
        visible_slots:
            Track slot indices that are active and visible this frame.
        slot_evidences:
            Mapping from slot_index → list of evidence items for that slot.
            May be empty for slots without evidence this frame.

        Returns
        -------
        list[IdentityAssignment]
            One assignment per visible slot, in the same order as
            *visible_slots*.
        """
        if not visible_slots:
            return []

        # Step 1+2: predict and fuse evidence for each visible slot
        for slot in visible_slots:
            blocked_labels = {
                other_belief.committed_label
                for other_slot, other_belief in self._beliefs.items()
                if other_slot != slot and other_belief.committed_label
            }
            belief = self._get_or_create_belief(
                slot,
                blocked_labels=blocked_labels,
                frame_idx=frame_idx,
            )
            self._predict_belief(belief)
            evs = slot_evidences.get(slot, [])
            belief.last_evidence_sources = tuple(
                sorted(
                    {str(ev.source_name) for ev in evs if str(ev.source_name).strip()}
                )
            )
            belief.last_conflict_flag = False
            self._fuse_evidence(belief, evs)
            belief.last_frame_idx = int(frame_idx)

        # Step 2.5: detect & execute swaps using pre-bias (raw evidence) posteriors
        # so that sustained mutual mismatch can override the slot-lock bias that
        # would otherwise pin both slots to wrong identities.
        self._detect_and_execute_swaps(visible_slots, frame_idx)

        # Step 3: apply lock bias (post-swap, so the bias follows the new
        # committed identity)
        for slot in visible_slots:
            self._apply_slot_lock_bias(self._beliefs[slot])

        # Step 4: uniqueness-constrained visible-slot assignment
        assigned_labels = self._solve_visible_assignment(visible_slots)

        # Step 5+6: update commitment and build outputs
        assignments: list[IdentityAssignment] = []
        for slot in visible_slots:
            belief = self._beliefs[slot]
            label = assigned_labels.get(slot)
            probs = self._posterior_probs(belief)
            known_probs = probs[1:]  # exclude unknown
            best_known_label = None
            best_known_conf = 0.0
            if len(known_probs) > 0:
                best_known_idx = int(np.argmax(known_probs)) + 1
                best_known_label = self._catalog.label_of(best_known_idx)
                best_known_conf = float(probs[best_known_idx])

            if label and self._catalog.contains(label):
                cat_idx = self._catalog.index_of(label)
            else:
                cat_idx = 0
            confidence = float(probs[cat_idx]) if cat_idx > 0 else 0.0

            ent = self._entropy(probs)
            if len(known_probs) >= 2:
                top2 = np.partition(known_probs, -2)[-2:]
                margin = float(top2[1] - top2[0])
            else:
                margin = 0.0

            belief.last_conflict_flag = bool(
                best_known_label
                and best_known_conf >= self._display_threshold
                and label != best_known_label
            )

            self._update_commitment(belief, label, confidence, frame_idx)

            # Report committed identity if committed, else best assignment
            out_label = belief.committed_label if belief.committed else label
            out_idx = belief.committed_index if belief.committed else cat_idx

            assignments.append(
                IdentityAssignment(
                    slot_index=slot,
                    label=out_label,
                    catalog_index=out_idx,
                    confidence=confidence,
                    entropy=ent,
                    margin=margin,
                    committed=belief.committed,
                )
            )

        return assignments

    def _solve_visible_assignment(
        self, visible_slots: list[int]
    ) -> dict[int, Optional[str]]:
        """Solve the partial injective assignment of slots to known identities.

        Uses the Hungarian algorithm when scipy is available, with dummy
        unassigned columns so slots can remain unassigned.
        Falls back to greedy argmax if scipy is absent.
        """
        if not visible_slots:
            return {}
        try:
            from scipy.optimize import linear_sum_assignment

            return self._hungarian_assignment(visible_slots, linear_sum_assignment)
        except ImportError:
            return self._greedy_assignment(visible_slots)

    def _hungarian_assignment(
        self,
        visible_slots: list[int],
        linear_sum_assignment,
    ) -> dict[int, Optional[str]]:
        """Hungarian-based uniqueness-constrained assignment."""
        N = len(visible_slots)
        K = self._catalog.num_known

        # N × (K + N) cost matrix: K identity columns + N dummy columns
        cost = np.zeros((N, K + N), dtype=np.float64)
        for i, slot in enumerate(visible_slots):
            belief = self._beliefs[slot]
            probs = self._posterior_probs(belief)
            # Known identity columns: cost = -log(prob)
            for j in range(K):
                cost[i, j] = -np.log(max(probs[j + 1], 1e-300))  # j+1 skips unknown
            # Dummy columns: cost = -log(unknown_prob)
            unknown_cost = -np.log(max(probs[0], 1e-300))
            for j in range(N):
                cost[i, K + j] = unknown_cost

        rows, cols = linear_sum_assignment(cost)
        result: dict[int, Optional[str]] = {}
        for r, c in zip(rows, cols):
            slot = visible_slots[r]
            if c < K:
                belief = self._beliefs[slot]
                probs = self._posterior_probs(belief)
                label_idx = c + 1  # skip unknown at 0
                if float(probs[label_idx]) >= self._display_threshold:
                    result[slot] = self._catalog.label_of(label_idx)
                else:
                    result[slot] = None
            else:
                result[slot] = None  # unassigned
        return result

    def _greedy_assignment(self, visible_slots: list[int]) -> dict[int, Optional[str]]:
        """Greedy argmax fallback (no uniqueness guarantee among low-confidence cases)."""
        result: dict[int, Optional[str]] = {}
        used: set[int] = set()
        for slot in visible_slots:
            belief = self._beliefs[slot]
            probs = self._posterior_probs(belief)
            known_probs = probs[1:]
            best_k = int(np.argmax(known_probs))  # 0-based over knowns
            best_idx = best_k + 1  # catalog index
            best_conf = float(probs[best_idx])
            if best_conf >= self._display_threshold and best_idx not in used:
                result[slot] = self._catalog.label_of(best_idx)
                used.add(best_idx)
            else:
                result[slot] = None
        return result

    def _update_commitment(
        self,
        belief: TrackIdentityBelief,
        label: Optional[str],
        confidence: float,
        frame_idx: int,
    ) -> None:
        """Update commitment state and soft slot-lock (Phase 2)."""
        # Block commit if this identity is already committed on any other slot
        for other_slot, other_belief in self._beliefs.items():
            if (
                other_slot != belief.slot_index
                and other_belief.committed_label == label
            ):
                belief.stable_count = (
                    0  # not converging — reset so lock doesn't fire early
                )
                return

        if not (
            label
            and confidence >= self._commit_threshold
            and belief.hit_count >= self._commit_min_hits
            and self._catalog.contains(label)
        ):
            belief.stable_count = 0
            return

        cat_idx = self._catalog.index_of(label)
        probs = self._posterior_probs(belief)

        if belief.slot_lock_label and belief.slot_lock_label != label:
            try:
                lock_idx = self._catalog.index_of(belief.slot_lock_label)
                lock_conf = float(probs[lock_idx])
            except (KeyError, IndexError):
                lock_conf = 0.0
            if confidence - lock_conf >= self._slot_lock_override_margin:
                log.debug(
                    "Slot %d released soft-lock '%s' for '%s' (override margin %.3f)",
                    belief.slot_index,
                    belief.slot_lock_label,
                    label,
                    confidence - lock_conf,
                )
                belief.slot_lock_label = None
                belief.slot_lock_strength = 0.0
                belief.slot_lock_frame = 0

        if belief.committed and belief.committed_label not in (None, label):
            try:
                committed_conf = float(probs[belief.committed_index])
            except IndexError:
                committed_conf = 0.0
            if confidence - committed_conf < self._slot_lock_override_margin:
                belief.stable_count = 0
                return
            log.debug(
                "Slot %d revised commitment '%s' -> '%s' (override margin %.3f)",
                belief.slot_index,
                belief.committed_label,
                label,
                confidence - committed_conf,
            )
            belief.slot_lock_label = None
            belief.slot_lock_strength = 0.0
            belief.slot_lock_frame = 0

        if not belief.committed or belief.committed_label != label:
            belief.committed = True
            belief.committed_label = label
            belief.committed_index = cat_idx
            belief.stable_count = 0
            log.debug(
                "Slot %d committed to '%s' (conf=%.3f, hits=%d)",
                belief.slot_index,
                label,
                confidence,
                belief.hit_count,
            )

        belief.stable_count += 1
        if (
            belief.stable_count >= self._slot_lock_min_frames
            and belief.slot_lock_label != label
        ):
            belief.slot_lock_label = label
            belief.slot_lock_strength = self._slot_lock_strength
            belief.slot_lock_frame = frame_idx
            log.debug(
                "Slot %d soft-locked to '%s' at frame %d",
                belief.slot_index,
                label,
                frame_idx,
            )

    def _detect_and_execute_swaps(
        self,
        visible_slots: list[int],
        frame_idx: int,
    ) -> None:
        """Detect sustained mutual identity disagreement between committed
        slot pairs and atomically swap their committed labels.

        This runs on raw post-fusion (pre-lock-bias) posteriors so the slot lock
        cannot mask a real swap.  Pairwise only — three-way cycles must resolve
        as two consecutive pairwise swaps.
        """
        if not self._swap_enabled:
            return

        # Snapshot committed slots and their pre-bias posteriors
        committed_visible: list[int] = []
        probs_by_slot: dict[int, np.ndarray] = {}
        for slot in visible_slots:
            belief = self._beliefs.get(slot)
            if belief is None or not belief.committed or not belief.committed_label:
                continue
            committed_visible.append(slot)
            probs_by_slot[slot] = self._posterior_probs(belief)

        if len(committed_visible) < 2:
            # Drop counters tied to absent/uncommitted slots
            self._swap_evidence = {
                k: v
                for k, v in self._swap_evidence.items()
                if k[0] in probs_by_slot and k[1] in probs_by_slot
            }
            return

        already_swapped: set[int] = set()
        seen_pairs: set[tuple[int, int]] = set()
        committed_visible.sort()
        for i in range(len(committed_visible)):
            for j in range(i + 1, len(committed_visible)):
                a = committed_visible[i]
                b = committed_visible[j]
                pair = (a, b)
                seen_pairs.add(pair)
                if a in already_swapped or b in already_swapped:
                    continue
                belief_a = self._beliefs[a]
                belief_b = self._beliefs[b]
                if self._is_mutual_mismatch(
                    belief_a, belief_b, probs_by_slot[a], probs_by_slot[b]
                ):
                    self._swap_evidence[pair] = self._swap_evidence.get(pair, 0) + 1
                    if self._swap_evidence[pair] >= self._swap_min_frames:
                        self._execute_swap(a, b, frame_idx)
                        already_swapped.add(a)
                        already_swapped.add(b)
                        self._swap_evidence.pop(pair, None)
                else:
                    self._swap_evidence.pop(pair, None)

        # Drop counters for pairs that didn't appear this frame (stale state)
        self._swap_evidence = {
            k: v for k, v in self._swap_evidence.items() if k in seen_pairs
        }

    def _is_mutual_mismatch(
        self,
        belief_a: TrackIdentityBelief,
        belief_b: TrackIdentityBelief,
        probs_a: np.ndarray,
        probs_b: np.ndarray,
    ) -> bool:
        """True if A's evidence favors B's identity and B's evidence favors A's,
        each by at least ``IDENTITY_SWAP_CONF_MARGIN`` and above the display
        threshold.
        """
        if belief_a.committed_label is None or belief_b.committed_label is None:
            return False
        if belief_a.committed_label == belief_b.committed_label:
            return False
        idx_a = int(belief_a.committed_index)
        idx_b = int(belief_b.committed_index)
        if idx_a <= 0 or idx_b <= 0:
            return False
        if idx_a >= probs_a.shape[0] or idx_b >= probs_a.shape[0]:
            return False

        margin = self._swap_conf_margin
        thresh = self._display_threshold
        a_likes_b = (
            float(probs_a[idx_b]) >= thresh
            and float(probs_a[idx_b]) - float(probs_a[idx_a]) >= margin
        )
        b_likes_a = (
            float(probs_b[idx_a]) >= thresh
            and float(probs_b[idx_a]) - float(probs_b[idx_b]) >= margin
        )
        return a_likes_b and b_likes_a

    def _execute_swap(self, slot_a: int, slot_b: int, frame_idx: int) -> None:
        """Atomically swap committed identities (and slot-lock state) between
        two beliefs.  Trajectory geometry is unchanged; only the identity
        labels exchange."""
        bel_a = self._beliefs[slot_a]
        bel_b = self._beliefs[slot_b]
        log.info(
            "Identity swap fired: slot %d ('%s') <-> slot %d ('%s') at frame %d",
            slot_a,
            bel_a.committed_label,
            slot_b,
            bel_b.committed_label,
            frame_idx,
        )
        bel_a.committed_label, bel_b.committed_label = (
            bel_b.committed_label,
            bel_a.committed_label,
        )
        bel_a.committed_index, bel_b.committed_index = (
            bel_b.committed_index,
            bel_a.committed_index,
        )
        bel_a.slot_lock_label, bel_b.slot_lock_label = (
            bel_b.slot_lock_label,
            bel_a.slot_lock_label,
        )
        bel_a.slot_lock_strength, bel_b.slot_lock_strength = (
            bel_b.slot_lock_strength,
            bel_a.slot_lock_strength,
        )
        bel_a.slot_lock_frame, bel_b.slot_lock_frame = (
            bel_b.slot_lock_frame,
            bel_a.slot_lock_frame,
        )
        bel_a.stable_count = 0
        bel_b.stable_count = 0

    def clear_slot(
        self,
        slot_index: int,
        reason: str = "",
        respawn_frame_idx: Optional[int] = None,
    ) -> None:
        """Clear belief state for *slot_index*.

        Called when a lost slot is respawned.  Logs an info message when the
        slot had a committed identity (potential identity change event).

        Parameters
        ----------
        slot_index:
            Slot to clear.
        reason:
            Human-readable description of why the slot was cleared (e.g.
            ``'respawn'``, ``'timeout'``).
        """
        if slot_index in self._beliefs:
            old = self._beliefs[slot_index]
            if respawn_frame_idx is not None and self._respawn_prior_strength > 0.0:
                self._respawn_priors[slot_index] = RespawnPrior(
                    slot_index=slot_index,
                    log_posterior=np.asarray(
                        old.log_posterior, dtype=np.float64
                    ).copy(),
                    committed_label=old.committed_label,
                    committed_index=int(old.committed_index),
                    last_frame_idx=int(old.last_frame_idx),
                )
            if old.committed and old.committed_label:
                new_reason = reason or "respawn"
                log.info(
                    "Slot %d (was committed to '%s') cleared: %s",
                    slot_index,
                    old.committed_label,
                    new_reason,
                )
            del self._beliefs[slot_index]

        # Drop any swap-evidence counters that referenced this slot
        if self._swap_evidence:
            self._swap_evidence = {
                k: v for k, v in self._swap_evidence.items() if slot_index not in k
            }

    def get_belief(self, slot_index: int) -> Optional[TrackIdentityBelief]:
        """Return the current belief for *slot_index*, or ``None`` if absent."""
        return self._beliefs.get(slot_index)

    def get_slot_log_posteriors(self, slots: list[int]) -> dict[int, np.ndarray]:
        """Return current log-posteriors for *slots* without modifying state.

        Used to build per-track priors for the Bayesian identity cost term in
        the assignment cost matrix.  Slots with no belief yet return a uniform
        prior over all known identities.
        """
        result: dict[int, np.ndarray] = {}
        for slot in slots:
            belief = self._beliefs.get(slot)
            if belief is not None:
                result[slot] = belief.log_posterior.copy()
            else:
                result[slot] = self._initial_log_posterior()
        return result

    def decay_absent_slot_beliefs(self, absent_slots: list[int]) -> None:
        """Apply the Markov transition to committed-but-absent slot beliefs.

        Called each frame for slots that are lost/reserved so their beliefs
        stay current (diffuse but not frozen) for identity-first rejoining.
        Only processes slots that have a committed_label — uncommitted absent
        slots don't need decay since they'll be cleared on respawn anyway.
        """
        for slot in absent_slots:
            belief = self._beliefs.get(slot)
            if belief is not None and belief.committed_label:
                self._predict_belief(belief)

    def all_active_slots(self) -> list[int]:
        """Return list of all slot indices with active beliefs."""
        return list(self._beliefs.keys())
