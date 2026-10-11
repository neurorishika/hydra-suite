"""ClusterDialog — configure clustering settings for ClassKit."""

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from hydra_suite.classkit.core.cluster.clustering_backend import (
    probe_clustering_backend,
)


class ClusterDialog(QDialog):
    """Dialog for configuring clustering."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cluster Configuration")
        self.setMinimumWidth(500)

        ann_probe = probe_clustering_backend()

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        header = QLabel("<h2>Clustering Settings</h2>")
        layout.addWidget(header)

        form = QFormLayout()

        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(10, 10000)
        self.n_clusters_spin.setValue(500)
        self.n_clusters_spin.setSuffix(" clusters")
        form.addRow("<b>Number of Clusters:</b>", self.n_clusters_spin)

        self.method_combo = QComboBox()
        self.method_combo.addItem("K-means (MiniBatch above 5,000 images)", "kmeans")
        self.method_combo.addItem("MiniBatch K-means (fastest)", "minibatch")
        if ann_probe.get("hdbscan_available"):
            self.method_combo.addItem("HDBSCAN (density-based)", "hdbscan")
        self.method_combo.setToolTip(
            "K-means is deterministic and recommended. HDBSCAN finds dense "
            "groups on its own; the cluster count then sets its minimum "
            "cluster size (images / clusters)."
        )
        form.addRow("<b>Method:</b>", self.method_combo)

        layout.addLayout(form)

        info_text = (
            "<b>Clustering Tips:</b><br>"
            + "\u2022 Use many clusters (100s-1000s) to capture visual modes<br>"
            + "\u2022 Clusters \u2260 classes (fine-grained structure)<br>"
            + f"\u2022 Neighbour search: <b>{ann_probe.get('best_ann', 'numpy')}</b>"
        )
        if ann_probe.get("hdbscan_available"):
            info_text += (
                "<br>\u2022 HDBSCAN (density-based) is available for advanced users."
            )

        info = QLabel(info_text)
        info.setWordWrap(True)
        info.setStyleSheet(
            "padding: 12px; background-color: #252526; border-radius: 6px; "
            + "border-left: 3px solid #0e639c; color: #ffffff; line-height: 1.8;"
        )
        layout.addWidget(info)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_settings(self):
        """Return ``(n_clusters, method)`` with method one of
        ``"kmeans"``, ``"minibatch"`` or ``"hdbscan"``."""
        return self.n_clusters_spin.value(), self.method_combo.currentData()
