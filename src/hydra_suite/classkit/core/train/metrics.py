"""
Classification metrics for evaluation and monitoring.
"""

try:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )
except ImportError:
    np = None
    confusion_matrix = None
    accuracy_score = None
    precision_recall_fscore_support = None

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ClassMetrics:
    """Per-class metrics."""

    class_id: int
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int  # number of samples


@dataclass
class EvalMetrics:
    """Overall evaluation metrics."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: List[ClassMetrics]
    confusion_matrix: np.ndarray
    num_samples: int


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> EvalMetrics:
    """
    Compute comprehensive classification metrics.

    Args:
        predictions: (N,) predicted labels
        labels: (N,) ground-truth labels
        class_names: Optional list of class names

    Returns:
        EvalMetrics object
    """
    if accuracy_score is None:
        raise ImportError("sklearn required for metrics")

    # Overall accuracy
    acc = accuracy_score(labels, predictions)

    # sklearn reports only the class ids that occur in labels or predictions, in
    # sorted order, so row ``i`` of every per-class array belongs to
    # ``present[i]``, not to ``class_names[i]``.  Indexing names by row position
    # silently relabels classes whenever any class is absent.
    present = np.unique(np.concatenate([np.asarray(labels), np.asarray(predictions)]))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    # Macro averages
    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f = f1.mean()

    # Weighted F1
    weighted_f = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )[2]

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Per-class details
    num_classes = len(precision)
    if class_names is None:
        class_names = [f"Class_{int(c)}" for c in present]

    per_class = []
    for i in range(num_classes):
        per_class.append(
            ClassMetrics(
                class_id=int(present[i]),
                class_name=(
                    class_names[int(present[i])]
                    if int(present[i]) < len(class_names)
                    else f"Class_{int(present[i])}"
                ),
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1=float(f1[i]),
                support=int(support[i]),
            )
        )

    return EvalMetrics(
        accuracy=float(acc),
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f),
        weighted_f1=float(weighted_f),
        per_class=per_class,
        confusion_matrix=cm,
        num_samples=len(labels),
    )


def format_metrics_report(metrics: EvalMetrics) -> str:
    """Format metrics as a readable text report."""
    lines = []
    lines.append("=" * 60)
    lines.append("Classification Metrics")
    lines.append("=" * 60)
    lines.append(f"Accuracy:        {metrics.accuracy:.3f}")
    lines.append(f"Macro Precision: {metrics.macro_precision:.3f}")
    lines.append(f"Macro Recall:    {metrics.macro_recall:.3f}")
    lines.append(f"Macro F1:        {metrics.macro_f1:.3f}")
    lines.append(f"Weighted F1:     {metrics.weighted_f1:.3f}")
    lines.append(f"Samples:         {metrics.num_samples}")
    lines.append("")
    lines.append("Per-Class Metrics:")
    lines.append("-" * 60)
    lines.append(
        f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    )
    lines.append("-" * 60)

    for cm in metrics.per_class:
        lines.append(
            f"{cm.class_name:<20} {cm.precision:>10.3f} {cm.recall:>10.3f} "
            f"{cm.f1:>10.3f} {cm.support:>10}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)


def compute_label_coverage(
    labels: np.ndarray, cluster_assignments: np.ndarray
) -> Dict[int, float]:
    """
    Compute fraction of labeled samples per cluster.

    Args:
        labels: (N,) labels (-1 for unlabeled)
        cluster_assignments: (N,) cluster IDs

    Returns:
        Dictionary: cluster_id -> fraction labeled
    """
    n_clusters = cluster_assignments.max() + 1
    coverage = {}

    for cluster_id in range(n_clusters):
        mask = cluster_assignments == cluster_id
        if mask.sum() == 0:
            coverage[cluster_id] = 0.0
            continue

        cluster_labels = labels[mask]
        labeled = (cluster_labels >= 0).sum()
        coverage[cluster_id] = labeled / len(cluster_labels)

    return coverage
