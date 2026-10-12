"""
Active learning batch acquisition strategies.

Implements the recipe (default fractions):
- 30% uncertainty (highest entropy / smallest margin)
- 25% diversity (k-center / farthest-first)
- 20% error neighbours (unlabeled images most similar to images the model gets
  wrong, round-robin over the errors, most confidently wrong first)
- 10% rare clusters (sampled with weight ``cluster_size ** -alpha``, so small
  clusters are likelier than big ones, and less so once labeled)
- 10% representativeness (dense clusters with low label coverage)
- 5% audits (random + high-disagreement clusters)

Slots run in that order against a shrinking pool, so no image is picked twice
and the batch fills to ``batch_size``.  A slot whose signal is unavailable
(no known errors, no clusters) hands its budget to uncertainty, so the batch
size is always honoured.
"""

try:
    import numpy as np
except ImportError:
    np = None

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .density import select_diverse_samples
from .combination import CombinationBalanceSelector
from .error_mining import (
    ErrorNeighborSelector,
    RareClusterSelector,
    find_prediction_errors,
)


@dataclass
class BatchConfig:
    """Configuration for batch acquisition."""

    batch_size: int = 100
    uncertainty_fraction: float = 0.25
    diversity_fraction: float = 0.20
    representative_fraction: float = 0.05
    combination_fraction: float = 0.15  # images likely to be missing / rare combos
    combination_power: float = 1.0  # weight = (count + 1) ** -power per combination
    error_fraction: float = 0.20  # unlabeled images similar to known model errors
    rare_cluster_fraction: float = 0.10  # bias toward small / under-labeled clusters
    rare_alpha: float = 0.75  # 0 = proportional to cluster size, 1 = equal per cluster
    error_same_pred_boost: float = (
        0.05  # cosine bonus for repeating the same wrong answer
    )
    audit_fraction: float = 0.05  # noqa: DC01  (dataclass field)
    per_cluster_cap: Optional[int] = None  # Max samples per cluster
    min_per_class: int = (
        0  # noqa: DC01  (dataclass field) — minimum per class if imbalanced
    )
    balance_mode: bool = (
        False  # noqa: DC01 — replace representative slot with label-balance scoring
    )


class UncertaintySelector:
    """Select samples with highest uncertainty."""

    @staticmethod
    def entropy(probs: np.ndarray) -> np.ndarray:
        """
        Compute entropy of probability distributions.

        Args:
            probs: (N, num_classes) probabilities

        Returns:
            entropy: (N,) entropy values
        """
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        return -(probs * np.log(probs)).sum(axis=1)

    @staticmethod
    def margin(probs: np.ndarray) -> np.ndarray:
        """
        Compute margin between top two predictions.

        Args:
            probs: (N, num_classes) probabilities

        Returns:
            margin: (N,) margin values (smaller = more uncertain)
        """
        # Sort probabilities
        sorted_probs = np.sort(probs, axis=1)
        # Margin = difference between top 2
        return sorted_probs[:, -1] - sorted_probs[:, -2]

    def select(
        self,
        probs: np.ndarray,
        n_samples: int,
        unlabeled_mask: np.ndarray,
        method: str = "entropy",
    ) -> np.ndarray:
        """
        Select most uncertain samples.

        Args:
            probs: (N, num_classes) prediction probabilities
            n_samples: Number to select
            unlabeled_mask: (N,) boolean mask of unlabeled samples
            method: 'entropy' or 'margin'

        Returns:
            selected_indices: (n_samples,) indices
        """
        if method == "entropy":
            uncertainty = self.entropy(probs)
        elif method == "margin":
            uncertainty = -self.margin(probs)  # Negative so higher is more uncertain
        else:
            raise ValueError(f"Unknown method: {method}")

        # Only consider unlabeled
        uncertainty[~unlabeled_mask] = -np.inf

        # Never hand back masked (already-labeled) rows when asked for more
        # samples than remain.
        n_samples = min(int(n_samples), int(np.count_nonzero(unlabeled_mask)))
        if n_samples <= 0:
            return np.array([], dtype=int)
        selected = np.argsort(uncertainty)[-n_samples:][::-1]
        return selected


class RepresentativeSelector:
    """Select samples from dense, under-labeled clusters."""

    def select(
        self,
        embeddings: np.ndarray,
        cluster_assignments: np.ndarray,
        label_coverage: Dict[int, float],
        cluster_densities: np.ndarray,
        n_samples: int,
        unlabeled_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Select samples from dense clusters with low label coverage.

        Args:
            embeddings: (N, D)
            cluster_assignments: (N,) cluster IDs
            label_coverage: cluster_id -> fraction labeled
            cluster_densities: (K,) density per cluster (lower = denser)
            n_samples: Number to select
            unlabeled_mask: (N,) boolean mask

        Returns:
            selected_indices: (n_samples,) indices
        """
        # Score clusters: dense + low coverage
        n_clusters = len(cluster_densities)
        cluster_scores = np.zeros(n_clusters)

        for cluster_id in range(n_clusters):
            coverage = label_coverage.get(cluster_id, 0.0)
            density = cluster_densities[cluster_id]

            # Higher score = denser + less labeled
            # Use inverse density (smaller distance = higher density)
            if density > 0:
                cluster_scores[cluster_id] = (1.0 - coverage) / (density + 1e-6)

        # Select samples from top clusters
        selected = []
        top_clusters = np.argsort(cluster_scores)[::-1]

        for cluster_id in top_clusters:
            if len(selected) >= n_samples:
                break

            # Get unlabeled samples from this cluster
            mask = (cluster_assignments == cluster_id) & unlabeled_mask
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            # Take random subset from this cluster
            needed = n_samples - len(selected)
            take = min(len(indices), needed)
            selected.extend(np.random.choice(indices, take, replace=False))

        return np.array(selected[:n_samples])


class BalanceSelector:
    """Select samples that reduce label-distribution imbalance in the labeled set.

    Score = Σ_c  prob[i,c] / (labeled_count[c] + 1)

    Samples predicted (softly) to belong to underrepresented classes score higher.
    """

    def select(
        self,
        probs: np.ndarray,
        class_names: List[str],
        image_labels: List[Optional[str]],
        n_samples: int,
        unlabeled_mask: np.ndarray,
    ) -> np.ndarray:
        from collections import Counter

        labeled_count: Counter = Counter(
            lbl for lbl in image_labels if lbl is not None and lbl != ""
        )

        # Build per-class inverse-frequency weight vector (shape C)
        weights = np.array(
            [1.0 / (labeled_count.get(name, 0) + 1) for name in class_names],
            dtype=np.float64,
        )

        # Soft balance score: probs (N,C) @ weights (C,) → (N,)
        scores = probs.astype(np.float64) @ weights
        scores[~unlabeled_mask] = -np.inf

        n_avail = int(unlabeled_mask.sum())
        k = min(n_samples, n_avail)
        if k <= 0:
            return np.array([], dtype=int)

        selected = np.argsort(scores)[-k:][::-1]
        return selected


class AuditSelector:
    """Select audit samples for quality assurance."""

    def select(
        self,
        n_samples: int,
        unlabeled_mask: np.ndarray,
        cluster_assignments: Optional[np.ndarray] = None,
        cluster_disagreements: Optional[Dict[int, float]] = None,
        audit_fraction_random: float = 0.5,
    ) -> np.ndarray:
        """
        Select audit samples.

        Args:
            n_samples: Number to select
            unlabeled_mask: (N,) boolean mask
            cluster_assignments: (N,) cluster IDs (optional)
            cluster_disagreements: cluster_id -> disagreement rate (optional)
            audit_fraction_random: Fraction of audits that are random

        Returns:
            selected_indices: (n_samples,) indices
        """
        n_random = int(n_samples * audit_fraction_random)
        n_targeted = n_samples - n_random

        selected = []

        # Random audits
        unlabeled_indices = np.where(unlabeled_mask)[0]
        if len(unlabeled_indices) > 0:
            random_selected = np.random.choice(
                unlabeled_indices, min(n_random, len(unlabeled_indices)), replace=False
            )
            selected.extend(random_selected)

        # Targeted audits (high-disagreement clusters)
        if (
            cluster_assignments is not None
            and cluster_disagreements is not None
            and n_targeted > 0
        ):
            # Sort clusters by disagreement
            sorted_clusters = sorted(
                cluster_disagreements.items(), key=lambda x: x[1], reverse=True
            )

            for cluster_id, _ in sorted_clusters:
                if len(selected) >= n_samples:
                    break

                mask = (cluster_assignments == cluster_id) & unlabeled_mask
                indices = np.where(mask)[0]

                if len(indices) == 0:
                    continue

                needed = n_samples - len(selected)
                take = min(len(indices), needed)
                selected.extend(np.random.choice(indices, take, replace=False))

        return np.array(selected[:n_samples])


class BatchAcquisition:
    """Main batch acquisition orchestrator."""

    def __init__(
        self, config: Optional[BatchConfig] = None, seed: Optional[int] = None
    ):
        self.config = config or BatchConfig()
        self.uncertainty_selector = UncertaintySelector()
        self.representative_selector = RepresentativeSelector()
        self.balance_selector = BalanceSelector()
        self.audit_selector = AuditSelector()
        self.error_selector = ErrorNeighborSelector(self.config.error_same_pred_boost)
        self.rare_selector = RareClusterSelector(self.config.rare_alpha)
        self.combination_selector = CombinationBalanceSelector(
            self.config.combination_power
        )
        self._rng = np.random.default_rng(seed)
        # Diagnostics for the most recent select_batch call (not part of the
        # (indices, breakdown) return so existing callers are unaffected).
        self.last_info: Dict[str, object] = {}

    def select_batch(
        self,
        embeddings: np.ndarray,
        probs: np.ndarray,
        unlabeled_mask: np.ndarray,
        cluster_assignments: Optional[np.ndarray] = None,
        label_coverage: Optional[Dict[int, float]] = None,
        cluster_densities: Optional[np.ndarray] = None,
        cluster_disagreements: Optional[Dict[int, float]] = None,
        image_labels: Optional[List[Optional[str]]] = None,
        class_names: Optional[List[str]] = None,
        predicted_labels: Optional[List[Optional[str]]] = None,
        prediction_confidence: Optional[np.ndarray] = None,
        trusted_label_mask: Optional[np.ndarray] = None,
        factor_probs: Optional[List[np.ndarray]] = None,
        factor_labels: Optional[List[List[str]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, List[int]]]:
        """
        Select a batch using the full recipe.

        Args:
            embeddings: (N, D) embeddings
            probs: (N, num_classes) prediction probabilities
            unlabeled_mask: (N,) boolean mask
            cluster_assignments: (N,) cluster IDs (optional; ``< 0`` = noise)
            label_coverage: cluster_id -> fraction labeled (optional)
            cluster_densities: (K,) density per cluster (optional)
            cluster_disagreements: cluster_id -> disagreement (optional)
            image_labels: per-sample label strings (balance mode and error mining)
            class_names: model output class names aligned with probs columns (optional)
            predicted_labels: per-sample predicted label strings; with
                ``image_labels`` this enables the error-neighbour slot.  Use the
                same (head-aware) strings the GUI shows, so multi-head models
                compare composite labels.
            prediction_confidence: (N,) confidence of each prediction; ranks how
                bad each error is (confidently wrong first)
            trusted_label_mask: (N,) True where ``image_labels`` is human-verified
                ground truth.  Unverified machine labels must be excluded or the
                model's own guesses would count as errors / ground truth.
            factor_probs / factor_labels: per-factor marginals ``[(N, L_f)]`` and
                the scheme's label list per factor; enable the combination slot,
                which targets under-represented (or never-seen) factor
                combinations.

        Returns:
            selected_indices: (batch_size,) selected indices
            breakdown: Dictionary with indices per reason
        """
        cfg = self.config
        breakdown: Dict[str, List[int]] = {}
        info: Dict[str, object] = {}
        self.last_info = info

        unlabeled_mask = np.asarray(unlabeled_mask, dtype=bool)
        avail = unlabeled_mask.copy()  # shrinks as slots claim images
        batch_size = min(int(cfg.batch_size), int(avail.sum()))

        fractions = {
            "error": max(0.0, cfg.error_fraction),
            "rare": max(0.0, cfg.rare_cluster_fraction),
            "combination": max(0.0, cfg.combination_fraction),
            "uncertainty": max(0.0, cfg.uncertainty_fraction),
            "diversity": max(0.0, cfg.diversity_fraction),
            "representative": max(0.0, cfg.representative_fraction),
        }
        total = sum(fractions.values())
        if total > 1.0:  # over-subscribed recipe: scale every slot down evenly
            fractions = {k: v / total for k, v in fractions.items()}
        n_error = int(batch_size * fractions["error"])
        n_rare = int(batch_size * fractions["rare"])
        n_combo = int(batch_size * fractions["combination"])
        n_uncertainty = int(batch_size * fractions["uncertainty"])
        n_diversity = int(batch_size * fractions["diversity"])
        n_representative = int(batch_size * fractions["representative"])
        n_audit = max(
            0,
            batch_size
            - (
                n_error
                + n_rare
                + n_combo
                + n_uncertainty
                + n_diversity
                + n_representative
            ),
        )

        chosen: List[int] = []

        def claim(reason: str, picked) -> None:
            picked = np.asarray(picked, dtype=int).reshape(-1)
            picked = picked[avail[picked]] if picked.size else picked
            if picked.size == 0:
                return
            avail[picked] = False
            chosen.extend(int(i) for i in picked)
            breakdown.setdefault(reason, []).extend(int(i) for i in picked)

        # 1. Error neighbours: unlabeled images that look like known failures.
        if n_error > 0 and image_labels is not None and predicted_labels is not None:
            errors = find_prediction_errors(
                image_labels,
                predicted_labels,
                prediction_confidence,
                trusted_label_mask,
            )
            info["n_errors"] = len(errors)
            if len(errors):
                res = self.error_selector.select(
                    embeddings, errors, avail, n_error, predicted_labels
                )
                claim("error_neighbors", res.indices)
                info["error_sources"] = {
                    int(i): int(res.source_error[int(i)])
                    for i in res.indices
                    if int(i) in res.source_error
                }
                info["error_pairs"] = {
                    int(i): (t, p)
                    for i, t, p in zip(
                        errors.indices, errors.true_labels, errors.pred_labels
                    )
                }

        # 1b. Combination balance: likely members of rare / missing combinations.
        if (
            n_combo > 0
            and factor_probs is not None
            and factor_labels is not None
            and image_labels is not None
        ):
            res = self.combination_selector.select(
                factor_probs,
                factor_labels,
                image_labels,
                avail,
                n_combo,
                trusted_label_mask,
            )
            claim("combination", res.indices)
            info["combination_targets"] = {
                int(i): c for i, c in res.target_combo.items()
            }
            info["combination_missing"] = list(res.missing)

        # 2. Uncertainty
        if n_uncertainty > 0:
            claim(
                "uncertainty",
                self.uncertainty_selector.select(probs, n_uncertainty, avail),
            )

        # 3. Rare clusters
        if n_rare > 0 and cluster_assignments is not None:
            picked = self.rare_selector.select(
                np.asarray(cluster_assignments),
                avail,
                n_rare,
                label_coverage,
                rng=self._rng,
            )
            claim("rare_cluster", picked)
            if picked.size:
                ids, counts = np.unique(
                    np.asarray(cluster_assignments)[picked], return_counts=True
                )
                info["rare_cluster_counts"] = {
                    int(i): int(c) for i, c in zip(ids, counts)
                }

        # 4. Diversity (k-center over what is still available)
        if n_diversity > 0 and avail.any():
            avail_idx = np.where(avail)[0]
            local = select_diverse_samples(embeddings[avail_idx], n_diversity)
            claim("diversity", avail_idx[np.asarray(local, dtype=int)])

        # 5. Representativeness / Balance
        if n_representative > 0 and avail.any():
            if (
                cfg.balance_mode
                and image_labels is not None
                and class_names is not None
            ):
                claim(
                    "balance",
                    self.balance_selector.select(
                        probs, class_names, image_labels, n_representative, avail
                    ),
                )
            elif cluster_assignments is not None:
                if label_coverage is None:
                    label_coverage = {}
                if cluster_densities is None:
                    from .density import compute_cluster_densities

                    cluster_densities = compute_cluster_densities(
                        embeddings, cluster_assignments
                    )
                claim(
                    "representative",
                    self.representative_selector.select(
                        embeddings,
                        cluster_assignments,
                        label_coverage,
                        cluster_densities,
                        n_representative,
                        avail,
                    ),
                )

        # 6. Audits
        if n_audit > 0 and avail.any():
            claim(
                "audit",
                self.audit_selector.select(
                    n_audit, avail, cluster_assignments, cluster_disagreements
                ),
            )

        # 7. Top-up: a slot with no signal (no errors yet, no clusters, or too
        # few candidates) hands its budget to uncertainty.
        short = batch_size - len(chosen)
        if short > 0 and avail.any():
            claim("uncertainty", self.uncertainty_selector.select(probs, short, avail))
            info["topped_up"] = short

        final_selected = np.asarray(chosen, dtype=int)

        # Apply per-cluster cap if specified
        if cfg.per_cluster_cap is not None and cluster_assignments is not None:
            final_selected = self._apply_cluster_cap(
                final_selected, cluster_assignments, cfg.per_cluster_cap
            )

        return final_selected, breakdown

    def _apply_cluster_cap(
        self, selected: np.ndarray, cluster_assignments: np.ndarray, cap: int
    ) -> np.ndarray:
        """Enforce maximum samples per cluster."""
        selected_clusters = cluster_assignments[selected]
        unique_clusters = np.unique(selected_clusters)

        capped = []
        for cluster_id in unique_clusters:
            mask = selected_clusters == cluster_id
            cluster_samples = selected[mask]

            if len(cluster_samples) > cap:
                # Random subsample
                cluster_samples = np.random.choice(cluster_samples, cap, replace=False)

            capped.extend(cluster_samples)

        return np.array(capped)
