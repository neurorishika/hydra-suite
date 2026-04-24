"""
Specialized worker classes for ClassKit background tasks.
"""

import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from ..core.export.splits import build_dataset_splits, build_label_expansion_split_key


class TaskSignals(QObject):
    """
    Signals for task progress and completion.
    """

    started = Signal()
    progress = Signal(int, str)  # (percentage, message)
    finished = Signal()
    success = Signal(object)  # result data
    error = Signal(str)  # error message


class IngestWorker(QRunnable):
    """Worker for ingesting images from folders/videos."""

    def __init__(
        self,
        source_path: Path,
        db_path: Path,
        project_classes: Optional[List[str]] = None,
        import_labels: bool = True,
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.source_path = source_path
        self.db_path = db_path
        self.project_classes = list(project_classes or [])
        self.import_labels = bool(import_labels)
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            self.signals.progress.emit(0, "Initializing ingestion...")

            from ..core.data.ingest import IngestWorker as Ingester
            from ..core.data.source_import import (
                build_source_import_plan,
                materialize_source_import_plan,
            )
            from ..core.store.db import ClassKitDB

            # Initialize
            self.signals.progress.emit(5, "Connecting to database...")
            db = ClassKitDB(self.db_path)
            ingester = Ingester(db)

            # Scan and ingest
            self.signals.progress.emit(10, f"Scanning folder: {self.source_path}...")
            plan = build_source_import_plan(self.source_path)
            self.signals.progress.emit(40, f"Found {len(plan.image_paths):,} images")

            project_labels = {
                str(label) for label in self.project_classes if str(label).strip()
            }
            imported_labels = {
                str(label) for label in plan.discovered_labels if str(label).strip()
            }
            if self.import_labels and imported_labels and project_labels:
                missing = sorted(imported_labels - project_labels)
                if missing:
                    raise ValueError(
                        "Imported dataset labels do not match the current project classes: "
                        + ", ".join(missing)
                    )

            plan = materialize_source_import_plan(
                plan,
                self.db_path,
                include_labels=self.import_labels,
            )
            image_paths = list(plan.image_paths)

            self.signals.progress.emit(50, "Computing image hashes...")
            ingester.ingest(image_paths, metadata_by_path=plan.metadata_by_path)

            if plan.label_updates:
                self.signals.progress.emit(80, "Applying imported labels...")
                db.update_labels_with_confidence_batch(
                    plan.label_updates,
                    label_source="import",
                    verified=True,
                    metadata_by_path=plan.metadata_by_path,
                )

            self.signals.progress.emit(90, "Writing to database...")

            self.signals.progress.emit(
                100, f"Complete! Ingested {len(image_paths):,} images"
            )
            self.signals.success.emit(
                {
                    "num_images": len(image_paths),
                    "source_kind": plan.source_kind,
                    "imported_labels": list(plan.discovered_labels),
                }
            )

        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class EmbeddingWorker(QRunnable):
    """Worker for computing embeddings."""

    def __init__(
        self,
        image_paths: List[Path],
        model_name: str,
        device: str,
        batch_size: int = 32,
        db_path: Optional[Path] = None,
        force_recompute: bool = False,
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.image_paths = image_paths
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.db_path = db_path
        self.force_recompute = force_recompute
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        db_cls = None
        reused_embeddings = None
        reused_metadata = None
        remaining_paths = list(self.image_paths)
        if self.db_path:
            from ..core.store.db import ClassKitDB as _ClassKitDB

            db_cls = _ClassKitDB

        from ..core.embed.embedder import (
            ModelLoadError,
            TimmEmbedder,
            resolve_embedder_device,
        )

        resolved_device = resolve_embedder_device(self.device)

        try:
            self.signals.started.emit()
            cache_model_name = self.model_name

            # Check for cached embeddings
            if db_cls is not None and not self.force_recompute:
                self.signals.progress.emit(0, "Checking for cached embeddings...")

                db = db_cls(self.db_path)

                cached = db.get_embeddings(cache_model_name, resolved_device)
                if cached is not None:
                    embeddings, metadata = cached
                    self.signals.progress.emit(
                        50, f"Found cached embeddings from {metadata['timestamp']}"
                    )
                    self.signals.progress.emit(
                        70,
                        f"Loaded {embeddings.shape[0]:,} embeddings (dim={embeddings.shape[1]})",
                    )
                    self.signals.progress.emit(100, "Using cached embeddings!")
                    self.signals.success.emit(
                        {
                            "embeddings": embeddings,
                            "dimension": embeddings.shape[1],
                            "cached": True,
                            "metadata": metadata,
                        }
                    )
                    return

                incremental = db.get_incremental_embedding_prefix(
                    cache_model_name,
                    resolved_device,
                    self.image_paths,
                )
                if incremental is not None:
                    reused_embeddings, reused_metadata, remaining_paths = incremental
                    self.signals.progress.emit(
                        12,
                        (
                            f"Reusing {reused_embeddings.shape[0]:,} cached embeddings; "
                            f"computing {len(remaining_paths):,} new"
                        ),
                    )
                else:
                    self.signals.progress.emit(
                        0, "No cache found, computing embeddings..."
                    )

            self.signals.progress.emit(0, f"Loading model: {self.model_name}...")

            # Create embedder
            self.signals.progress.emit(5, f"Creating embedder on {resolved_device}...")
            embedder = TimmEmbedder(model_name=self.model_name, device=resolved_device)

            self.signals.progress.emit(8, "Loading model weights...")
            embedder.load_model()

            self.signals.progress.emit(
                10,
                (
                    f"Computing embeddings for {len(remaining_paths):,} images..."
                    if reused_embeddings is not None
                    else f"Computing embeddings for {len(self.image_paths):,} images..."
                ),
            )
            self.signals.progress.emit(
                12, f"Batch size: {self.batch_size}, Device: {embedder.device}"
            )

            # Compute embeddings
            if reused_embeddings is not None:
                import numpy as np

                new_embeddings = embedder.embed(
                    remaining_paths,
                    batch_size=self.batch_size,
                )
                embeddings = np.concatenate(
                    [reused_embeddings, new_embeddings],
                    axis=0,
                )
            else:
                embeddings = embedder.embed(
                    self.image_paths,
                    batch_size=self.batch_size,
                )

            self.signals.progress.emit(
                90, f"Computed {embeddings.shape[0]:,} embeddings"
            )

            # Save to cache
            if db_cls is not None:
                self.signals.progress.emit(92, "Saving embeddings to cache...")
                db = db_cls(self.db_path)
                metadata = {
                    "model_name": self.model_name,
                    "reused_prefix_count": (
                        int(reused_embeddings.shape[0])
                        if reused_embeddings is not None
                        else 0
                    ),
                }
                if (
                    reused_metadata is not None
                    and reused_metadata.get("id") is not None
                ):
                    metadata["parent_embedding_id"] = int(reused_metadata["id"])

                embedding_cache_id = db.save_embeddings(
                    embeddings,
                    cache_model_name,
                    embedder.device,
                    self.batch_size,
                    meta=metadata,
                    image_paths=self.image_paths,
                )
                self.signals.progress.emit(95, "Cached for future use")
            else:
                embedding_cache_id = None

            self.signals.progress.emit(100, f"Complete! Shape: {embeddings.shape}")
            self.signals.success.emit(
                {
                    "embeddings": embeddings,
                    "dimension": embedder.dimension,
                    "cached": False,
                    "metadata": {
                        "id": embedding_cache_id,
                        "model_name": self.model_name,
                        "reused_prefix_count": (
                            int(reused_embeddings.shape[0])
                            if reused_embeddings is not None
                            else 0
                        ),
                    },
                }
            )

        except (ImportError, ModelLoadError) as e:
            self.signals.error.emit(str(e))
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class ClusteringWorker(QRunnable):
    """Worker for clustering embeddings."""

    def __init__(self, embeddings, n_clusters: int = 500, method: str = "minibatch"):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.embeddings = embeddings
        self.n_clusters = n_clusters
        self.method = method
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            self.signals.progress.emit(0, "Starting clustering pipeline...")
            self.signals.progress.emit(
                2,
                f"Input: {self.embeddings.shape[0]:,} vectors, dim={self.embeddings.shape[1]}",
            )
            self.signals.progress.emit(
                4, f"Target: {self.n_clusters} clusters  method={self.method}"
            )

            import numpy as np

            from ..core.cluster.clusterer import SKLearnClusterer

            self.signals.progress.emit(8, f"Initializing {self.method} clusterer...")
            clusterer = SKLearnClusterer(
                n_clusters=self.n_clusters,
                method=self.method,
                verbose=False,
            )

            self.signals.progress.emit(
                20,
                f"Fitting {self.embeddings.shape[0]:,} samples...",
            )
            assignments = clusterer.fit(self.embeddings)

            self.signals.progress.emit(75, "Computing cluster statistics...")
            stats = clusterer.compute_cluster_stats(self.embeddings, assignments)

            n_actual = len(np.unique(assignments))
            self.signals.progress.emit(
                90, f"Created {n_actual} clusters (requested {self.n_clusters})"
            )
            self.signals.progress.emit(100, "Clustering complete!")

            self.signals.success.emit(
                {
                    "assignments": assignments,
                    "centers": clusterer.cluster_centers,
                    "stats": stats,
                    "method": self.method,
                }
            )

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class UMAPWorker(QRunnable):
    """Worker for computing UMAP projection."""

    def __init__(self, embeddings, n_neighbors: int = 15, min_dist: float = 0.1):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.embeddings = embeddings
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            self.signals.progress.emit(0, "Initializing UMAP...")
            self.signals.progress.emit(
                5,
                f"Input: {self.embeddings.shape[0]:,} points, {self.embeddings.shape[1]} dims",
            )

            from ..core.viz.umap_reduce import UMAPReducer

            self.signals.progress.emit(
                10,
                f"Parameters: n_neighbors={self.n_neighbors}, min_dist={self.min_dist}",
            )
            reducer = UMAPReducer(n_neighbors=self.n_neighbors, min_dist=self.min_dist)

            self.signals.progress.emit(
                15, "Computing UMAP projection (this may take a while)..."
            )
            coords = reducer.fit_transform(self.embeddings)

            self.signals.progress.emit(95, f"Generated 2D coordinates: {coords.shape}")
            self.signals.progress.emit(100, "Complete! UMAP projection ready")
            self.signals.success.emit({"coords": coords})

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class TrainingWorker(QRunnable):
    """Worker for training classifier."""

    def __init__(
        self,
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        num_classes: int,
        model_type: str = "linear",
        device: str = "cpu",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        batch_size: int = 256,
        epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        early_stop_patience: int = 10,
        calibrate: bool = True,
    ):
        super().__init__()
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.val_embeddings = val_embeddings
        self.val_labels = val_labels
        self.num_classes = num_classes
        self.model_type = model_type
        self.device = device
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stop_patience = early_stop_patience
        self.calibrate = calibrate
        self.signals = TaskSignals()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            self.signals.progress.emit(0, "Initializing trainer...")

            from ..core.train.trainer import EmbeddingHeadTrainer

            input_dim = self.train_embeddings.shape[1]
            trainer = EmbeddingHeadTrainer(
                model_type=self.model_type,
                input_dim=input_dim,
                num_classes=self.num_classes,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
                device=self.device,
            )

            self.signals.progress.emit(10, "Training...")

            history = trainer.fit(
                self.train_embeddings,
                self.train_labels,
                self.val_embeddings,
                self.val_labels,
                batch_size=self.batch_size,
                epochs=self.epochs,
                lr=self.lr,
                weight_decay=self.weight_decay,
                early_stop_patience=self.early_stop_patience,
                verbose=False,
            )

            self.signals.progress.emit(80, "Calibrating...")

            metrics = None
            if self.val_embeddings is not None:
                if self.calibrate:
                    trainer.calibrate(self.val_embeddings, self.val_labels)

                # Compute metrics on validation set
                metrics = trainer.evaluate(
                    self.val_embeddings, self.val_labels, calibrated=self.calibrate
                )

            self.signals.progress.emit(100, "Complete!")
            self.signals.success.emit(
                {"trainer": trainer, "history": history, "metrics": metrics}
            )

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class ClassKitTrainingWorker(QRunnable):
    """Worker for ClassKit classification training (flat or multi-head)."""

    def __init__(
        self,
        *,
        role,
        specs,
        run_dir: str,
        multi_head: bool = False,
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.role = role
        self.specs = (
            specs  # list of TrainingRunSpec (one per factor if multi_head, else one)
        )
        self.run_dir = run_dir
        self.multi_head = multi_head
        self.signals = TaskSignals()
        self._canceled = False

    def cancel(self):
        self._canceled = True

    def run(self):
        from pathlib import Path

        from ...training.runner import run_training

        try:
            self.signals.started.emit()
            results = []

            for i, spec in enumerate(self.specs):
                if self._canceled:
                    self.signals.error.emit("Canceled by user")
                    return

                factor_run_dir = (
                    Path(self.run_dir) / f"factor_{i}"
                    if self.multi_head
                    else Path(self.run_dir) / "run"
                )
                n_specs = len(self.specs)

                def log_cb(msg: str, _i: int = i) -> None:
                    prefix = f"[factor {_i}] " if self.multi_head else ""
                    self.signals.progress.emit(-1, f"{prefix}{msg}")

                def progress_cb(
                    current: int, total: int, _i: int = i, _n: int = n_specs
                ) -> None:
                    total = max(1, int(total))
                    current = max(0, min(int(current), total))
                    completed_fraction = (
                        float(_i) + (float(current) / float(total))
                    ) / max(1.0, float(_n))
                    overall = int(round(completed_fraction * 100.0))
                    overall = max(0, min(100, overall))
                    self.signals.progress.emit(overall, "")

                result = run_training(
                    spec,
                    factor_run_dir,
                    log_cb=log_cb,
                    progress_cb=progress_cb,
                    should_cancel=lambda: self._canceled,
                )
                results.append(result)

                if not result.get("success"):
                    self.signals.error.emit(f"Training failed for spec {i}")
                    return

            self.signals.progress.emit(100, "")
            self.signals.success.emit(results)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class ExportWorker(QRunnable):
    """Worker for exporting datasets."""

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        output_path: Path,
        format: str = "imagefolder",
        class_names: Optional[Dict[int, str]] = None,
        split_strategy: str = "stratified",
        val_fraction: float = 0.2,
        test_fraction: float = 0.0,
        copy_files: bool = True,
        temp_dir: Optional[Path] = None,
        label_expansion: Optional[Dict[str, Dict[str, str]]] = None,
        force_monochrome: bool = False,
        preset_splits_by_path: Optional[Dict[str, str]] = None,
        preset_expanded_splits_by_key: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.image_paths = image_paths
        self.labels = labels
        self.output_path = output_path
        self.format = format
        self.class_names = class_names or {}
        self.split_strategy = str(split_strategy or "stratified")
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.copy_files = copy_files
        self.temp_dir = temp_dir
        # label_expansion: {"fliplr": {"left": "right", "right": "left"}, ...}
        self.label_expansion: Dict[str, Dict[str, str]] = label_expansion or {}
        self.force_monochrome = bool(force_monochrome)
        self.preset_splits_by_path = {
            str(Path(path).resolve()): str(split)
            for path, split in (preset_splits_by_path or {}).items()
        }
        self.preset_expanded_splits_by_key = {
            str(key): str(split)
            for key, split in (preset_expanded_splits_by_key or {}).items()
        }
        self.signals = TaskSignals()

    def _class_name(self, label: int) -> str:
        """Resolve class display name for a label ID."""
        return self.class_names.get(label, f"class_{label}")

    def _build_splits(self, labels: Optional[List[int]] = None):
        """Build deterministic train/val/test split assignments."""
        labels = list(self.labels if labels is None else labels)
        return build_dataset_splits(
            labels,
            strategy=self.split_strategy,
            val_fraction=self.val_fraction,
            test_fraction=self.test_fraction,
        )

    def _prepare_export_workspace(self) -> None:
        """Create a temporary export workspace when helper-generated images are needed."""
        if (self.label_expansion or self.force_monochrome) and self.temp_dir is None:
            import tempfile

            self._expansion_tmpdir = tempfile.mkdtemp(prefix="classkit_exp_")
            self.temp_dir = Path(self._expansion_tmpdir)
        else:
            self._expansion_tmpdir = None

    def _collect_valid_labels(
        self,
    ) -> tuple[List[Path], List[int], List[str], Dict[int, str]]:
        """Return filtered image/label/split data for labeled samples only."""
        valid = [
            (img_path, int(label))
            for img_path, label in zip(self.image_paths, self.labels)
            if int(label) >= 0
        ]
        if not valid:
            raise RuntimeError("No labeled samples found to export.")

        image_paths = [item[0] for item in valid]
        labels = [item[1] for item in valid]
        if self.preset_splits_by_path:
            splits = []
            unmatched = 0
            for path in image_paths:
                key = str(Path(path).resolve())
                split = self.preset_splits_by_path.get(key)
                if split is None:
                    unmatched += 1
                    split = "train"
                splits.append(split)
            if unmatched:
                self.signals.progress.emit(
                    -1,
                    f"WARNING: {unmatched} of {len(image_paths)} images did not match "
                    f"any preset split key and defaulted to 'train'. "
                    f"This may indicate a path resolution mismatch.",
                )
        else:
            splits = self._build_splits(labels)
        class_names = {
            int(label): str(name)
            for label, name in (self.class_names or {}).items()
            if int(label) >= 0
        }
        for label in set(labels):
            class_names.setdefault(int(label), self._class_name(int(label)))
        return image_paths, labels, splits, class_names

    @staticmethod
    def _resolve_expanded_label(
        mapping: Dict[str, str],
        src_name: str,
        name_to_int: Dict[str, int],
        name_to_int_ci: Dict[str, int],
    ) -> tuple[bool, Optional[int]]:
        """Resolve one expansion rule to a destination label and match status."""
        dst_name = mapping.get(src_name)
        if dst_name is None:
            src_name_ci = str(src_name).strip().lower()
            for key, value in mapping.items():
                if str(key).strip().lower() == src_name_ci:
                    dst_name = value
                    break
        if dst_name is None:
            return False, None

        dst_name_clean = str(dst_name).strip()
        dst_int = name_to_int.get(dst_name_clean)
        if dst_int is None:
            dst_int = name_to_int_ci.get(dst_name_clean.lower())
        return True, dst_int

    def _apply_label_expansion(
        self,
        image_paths: List[Path],
        labels: List[int],
        splits: List[str],
        class_names: Dict[int, str],
    ) -> tuple[List[Path], List[int], List[str]]:
        """Materialize deterministic flip-based label expansion for training samples."""
        if not self.label_expansion or not self.temp_dir:
            return image_paths, labels, splits

        import cv2

        name_to_int = {v: k for k, v in class_names.items()}
        name_to_int_ci = {str(v).strip().lower(): k for k, v in class_names.items()}
        flip_code = {"fliplr": 1, "flipud": 0}
        exp_dir = Path(self.temp_dir) / "label_expansion"
        exp_dir.mkdir(parents=True, exist_ok=True)

        extra_paths: List[Path] = []
        extra_labels: List[int] = []
        extra_splits: List[str] = []
        n_exp = 0
        n_skipped_unknown_dst = 0

        for axis, mapping in self.label_expansion.items():
            code = flip_code.get(axis)
            if code is None:
                continue
            axis_paths, axis_labels, axis_splits, skipped = self._expand_label_axis(
                axis,
                mapping,
                code,
                exp_dir,
                image_paths,
                labels,
                splits,
                class_names,
                name_to_int,
                name_to_int_ci,
                n_exp,
                cv2,
            )
            extra_paths.extend(axis_paths)
            extra_labels.extend(axis_labels)
            extra_splits.extend(axis_splits)
            n_skipped_unknown_dst += skipped
            n_exp += len(axis_paths)

        if n_exp:
            self.signals.progress.emit(55, f"Added {n_exp} label-expansion copies")
            self.copy_files = True
        if n_skipped_unknown_dst:
            self.signals.progress.emit(
                58,
                f"Skipped {n_skipped_unknown_dst} expansion rows with unknown destination class",
            )

        return (
            image_paths + extra_paths,
            labels + extra_labels,
            splits + extra_splits,
        )

    def _apply_monochrome_mode(
        self,
        image_paths: List[Path],
        labels: List[int],
        splits: List[str],
    ) -> tuple[List[Path], List[int], List[str]]:
        """Materialize grayscale RGB copies of the derived dataset when requested."""
        if not self.force_monochrome or not self.temp_dir:
            return image_paths, labels, splits

        import cv2

        mono_dir = Path(self.temp_dir) / "monochrome"
        mono_dir.mkdir(parents=True, exist_ok=True)

        converted_paths: List[Path] = []
        total = len(image_paths)
        for index, img_path in enumerate(image_paths):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(
                    f"Failed to read image for monochrome export: {img_path}"
                )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mono = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            out_p = mono_dir / f"mono_{index}_{img_path.stem}{img_path.suffix}"
            cv2.imwrite(str(out_p), mono)
            converted_paths.append(out_p)

        if total:
            self.signals.progress.emit(60, f"Converted {total} images to monochrome")
            self.copy_files = True
        return converted_paths, labels, splits

    def _expand_label_axis(
        self,
        axis: str,
        mapping: Dict[str, str],
        code: int,
        exp_dir: Path,
        image_paths: List[Path],
        labels: List[int],
        splits: List[str],
        class_names: Dict[int, str],
        name_to_int: Dict[str, int],
        name_to_int_ci: Dict[str, int],
        start_index: int,
        cv2_module,
    ) -> tuple[List[Path], List[int], int]:
        """Expand one configured axis and return new samples plus skipped count."""
        extra_paths: List[Path] = []
        extra_labels: List[int] = []
        extra_splits: List[str] = []
        n_skipped_unknown_dst = 0
        n_exp = start_index

        for img_path, label_int, split in zip(image_paths, labels, splits):
            matched_rule, dst_int = self._resolve_expanded_label(
                mapping,
                class_names.get(label_int, ""),
                name_to_int,
                name_to_int_ci,
            )
            if not matched_rule:
                continue
            if dst_int is None:
                n_skipped_unknown_dst += 1
                continue
            dst_name = class_names.get(dst_int, "")
            planned_split = None
            if self.preset_expanded_splits_by_key:
                planned_split = self.preset_expanded_splits_by_key.get(
                    build_label_expansion_split_key(img_path, axis, dst_name)
                )
            if planned_split is None and split != "train":
                continue

            img = cv2_module.imread(str(img_path), cv2_module.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(
                    f"Failed to read image for label expansion: {img_path}"
                )
            flipped = cv2_module.flip(img, code)
            stem = f"expn_{axis}_{n_exp}_{img_path.stem}"
            out_p = exp_dir / (stem + img_path.suffix)
            cv2_module.imwrite(str(out_p), flipped)
            extra_paths.append(out_p)
            extra_labels.append(dst_int)
            extra_splits.append(str(planned_split or "train"))
            n_exp += 1

        return extra_paths, extra_labels, extra_splits, n_skipped_unknown_dst

    def _export_imagefolder(
        self,
        image_paths: List[Path],
        labels: List[int],
        splits: List[str],
        class_names: Dict[int, str],
    ) -> None:
        from ..core.export.imagefolder import export_to_imagefolder

        records = [
            (path, class_names[label], split)
            for path, label, split in zip(image_paths, labels, splits)
        ]
        export_to_imagefolder(
            dataset_root=self.output_path,
            images=records,
            copy=self.copy_files,
        )

    def _export_csv(
        self,
        image_paths: List[Path],
        labels: List[int],
        splits: List[str],
        class_names: Dict[int, str],
    ) -> None:
        from ..core.export.parquet_csv import export_to_csv

        csv_path = self.output_path
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path / "labels.csv"
        export_to_csv(
            output_path=csv_path,
            image_paths=image_paths,
            labels=labels,
            class_names=class_names,
            splits=splits,
            include_header=True,
        )
        self.output_path = csv_path

    def _export_parquet(
        self,
        image_paths: List[Path],
        labels: List[int],
        splits: List[str],
        class_names: Dict[int, str],
    ) -> None:
        from ..core.export.parquet_csv import export_to_parquet

        parquet_path = self.output_path
        if parquet_path.suffix.lower() != ".parquet":
            parquet_path = parquet_path / "labels.parquet"
        export_to_parquet(
            output_path=parquet_path,
            image_paths=image_paths,
            labels=labels,
            class_names=class_names,
            splits=splits,
        )
        self.output_path = parquet_path

    def _export_ultralytics(
        self,
        image_paths: List[Path],
        labels: List[int],
        splits: List[str],
        class_names: Dict[int, str],
    ) -> None:
        from ..core.export.ultralytics_classify import export_ultralytics_classify

        train_images, train_labels = [], []
        val_images, val_labels = [], []
        test_images, test_labels = [], []
        for path, label, split in zip(image_paths, labels, splits):
            if split == "val":
                val_images.append(path)
                val_labels.append(label)
            elif split == "test":
                test_images.append(path)
                test_labels.append(label)
            else:
                train_images.append(path)
                train_labels.append(label)

        export_ultralytics_classify(
            output_path=self.output_path,
            train_images=train_images,
            train_labels=train_labels,
            val_images=val_images,
            val_labels=val_labels,
            test_images=test_images if test_images else None,
            test_labels=test_labels if test_labels else None,
            class_names=class_names,
            copy=self.copy_files,
        )

    def _export_dataset(
        self,
        image_paths: List[Path],
        labels: List[int],
        splits: List[str],
        class_names: Dict[int, str],
    ) -> None:
        """Dispatch to the requested dataset export format."""
        exporters = {
            "imagefolder": self._export_imagefolder,
            "csv": self._export_csv,
            "parquet": self._export_parquet,
            "ultralytics": self._export_ultralytics,
        }
        exporter = exporters.get(self.format)
        if exporter is None:
            raise ValueError(f"Unsupported export format: {self.format}")
        exporter(image_paths, labels, splits, class_names)

    @Slot()
    def run(self):
        try:
            import time as _time

            _t0 = _time.monotonic()
            self.signals.started.emit()
            self.signals.progress.emit(0, f"Exporting to {self.format}...")

            self._prepare_export_workspace()
            image_paths, labels, splits, class_names = self._collect_valid_labels()

            image_paths, labels, splits = self._apply_label_expansion(
                image_paths, labels, splits, class_names
            )
            image_paths, labels, splits = self._apply_monochrome_mode(
                image_paths, labels, splits
            )

            # Log per-class per-split statistics after deterministic expansion so
            # the displayed split counts match the exported dataset.
            from collections import Counter

            split_class_counts: dict = {}
            for split, label in zip(splits, labels):
                split_class_counts.setdefault(split, Counter())[label] += 1
            for split_name in ("train", "val", "test"):
                counts = split_class_counts.get(split_name)
                if not counts:
                    continue
                total = sum(counts.values())
                breakdown = ", ".join(
                    f"{class_names.get(lbl, f'class_{lbl}')}={n}"
                    for lbl, n in sorted(counts.items())
                )
                self.signals.progress.emit(
                    -1, f"Split {split_name}: {total} images ({breakdown})"
                )

            self.signals.progress.emit(
                -1, f"Exporting {len(image_paths)} images to {self.output_path}"
            )
            self._export_dataset(image_paths, labels, splits, class_names)

            _elapsed = _time.monotonic() - _t0
            self.signals.progress.emit(
                100, f"Export complete — {len(image_paths)} images in {_elapsed:.1f}s"
            )
            self.signals.success.emit(
                {
                    "output_path": self.output_path,
                    "format": self.format,
                    "num_exported": len(image_paths),
                    "num_classes": len(set(labels)),
                    "source_split_by_path": {
                        str(Path(path).resolve()): split
                        for path, split in zip(image_paths, splits)
                    },
                }
            )

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()
            # Clean up auto-created expansion temp dir (if we made it)
            if getattr(self, "_expansion_tmpdir", None):
                import shutil
                from contextlib import suppress

                with suppress(Exception):
                    shutil.rmtree(self._expansion_tmpdir, ignore_errors=True)
                self._expansion_tmpdir = None


class YoloInferenceWorker(QRunnable):
    """Run YOLO classification inference on all images, return per-image probabilities."""

    def __init__(
        self,
        model_path: Path,
        image_paths: List[Path],
        compute_runtime: str = "cpu",
        batch_size: int = 64,
        force_monochrome: bool = False,
        # Deprecated — use compute_runtime instead.
        device: str = "",
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.model_path = Path(model_path)
        self.image_paths = list(image_paths)
        if (not compute_runtime or compute_runtime == "cpu") and device not in (
            "",
            "cpu",
        ):
            compute_runtime = device
        self.compute_runtime = str(compute_runtime or "cpu")
        self.batch_size = batch_size
        self.force_monochrome = bool(force_monochrome)
        self.signals = TaskSignals()

    @staticmethod
    def _torch_device(rt: str) -> str:
        if rt in ("cuda", "onnx_cuda", "tensorrt"):
            return "cuda"
        if rt in ("mps", "onnx_coreml"):
            return "mps"
        if rt in ("rocm", "onnx_rocm"):
            return "cuda"  # kept for legacy configs; ROCm is no longer supported
        return "cpu"

    @staticmethod
    def _validate_model_suffix(model_path: Path) -> None:
        model_suffix = model_path.suffix.lower()
        if model_suffix == ".pth":
            raise ValueError(
                f"YoloInferenceWorker requires a YOLO classify artifact (.pt/.onnx/.engine/.trt); got {model_path.name}. "
                "Tiny CNN (.pth) models are not supported here."
            )
        if model_suffix not in {".pt", ".onnx", ".engine", ".trt"}:
            raise ValueError(
                f"Unsupported YOLO model artifact for inference: {model_path.name}"
            )

    def _export_runtime_artifact(self, rt: str, model_path: Path):
        from ultralytics import YOLO

        use_onnx = rt.startswith("onnx_")
        artifact_suffix = ".onnx" if use_onnx else ".engine"
        artifact_path = model_path.with_name(
            f"{model_path.stem}_classify_b1{artifact_suffix}"
        )
        needs_export = (not artifact_path.exists()) or (
            artifact_path.stat().st_mtime_ns < model_path.stat().st_mtime_ns
        )
        if not needs_export:
            return artifact_path

        self.signals.progress.emit(
            1,
            f"Preparing YOLO classify {artifact_suffix} runtime artifact...",
        )
        export_model = YOLO(str(model_path), task="classify")
        if use_onnx:
            export_path = export_model.export(
                format="onnx",
                dynamic=False,
                simplify=False,
                opset=17,
                batch=1,
                verbose=False,
            )
        else:
            export_device = self._torch_device(rt)
            export_model.to(export_device)
            export_path = export_model.export(
                format="engine",
                device=export_device,
                half=True,
                workspace=4,
                dynamic=False,
                batch=1,
                verbose=False,
            )
        exported = Path(export_path).expanduser().resolve()
        if not exported.exists():
            raise RuntimeError(f"YOLO runtime export output missing: {exported}")
        if exported != artifact_path:
            shutil.copy2(str(exported), str(artifact_path))
        return artifact_path

    def _resolve_model_path(self, rt: str) -> Path:
        model_path = self.model_path
        self._validate_model_suffix(model_path)
        use_exported_runtime = rt.startswith("onnx_") or rt == "tensorrt"
        if model_path.suffix.lower() == ".pt" and use_exported_runtime:
            model_path = self._export_runtime_artifact(rt, model_path)
        self._validate_model_suffix(model_path)
        return model_path

    def _load_model(self, rt: str, model_path: Path):
        from ultralytics import YOLO

        self.signals.progress.emit(
            0, f"Loading YOLO model ({rt}): {model_path.name}..."
        )
        model = YOLO(str(model_path), task="classify")
        class_names = (
            [model.names[i] for i in sorted(model.names.keys())]
            if hasattr(model, "names")
            else []
        )
        predict_device = None
        if model_path.suffix.lower() == ".pt":
            predict_device = self._torch_device(rt)
            try:
                model.to(predict_device)
                predict_device = None
            except Exception:
                pass
        return model, class_names, predict_device

    @staticmethod
    def _prepare_batch_input(batch_paths, force_monochrome=False):
        if not force_monochrome:
            return [str(path) for path in batch_paths]

        import cv2
        import numpy as np

        batch_input = []
        for path in batch_paths:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                batch_input.append(np.zeros((8, 8, 3), dtype=np.uint8))
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            batch_input.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        return batch_input

    def _run_batches(self, model, class_names, predict_device):
        import numpy as np

        num_images = len(self.image_paths)
        self.signals.progress.emit(
            5, f"Running inference on {num_images:,} images ({self.compute_runtime})..."
        )
        all_probs = []
        for batch_start in range(0, num_images, self.batch_size):
            batch_paths = self.image_paths[batch_start : batch_start + self.batch_size]
            batch_input = self._prepare_batch_input(
                batch_paths,
                force_monochrome=self.force_monochrome,
            )
            kwargs = {"verbose": False}
            if predict_device is not None:
                kwargs["device"] = predict_device
            results = model(batch_input, **kwargs)
            for result in results:
                if result.probs is not None:
                    all_probs.append(result.probs.data.cpu().numpy())
                else:
                    n_cls = max(len(class_names), 1)
                    all_probs.append(np.ones(n_cls) / n_cls)
            done = batch_start + len(batch_paths)
            pct = min(95, 5 + int(90 * done / num_images))
            self.signals.progress.emit(pct, f"Processed {done:,}/{num_images:,}")
        return np.array(all_probs)

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()

            from ...runtime.compute_runtime import _normalize_runtime

            rt = _normalize_runtime(self.compute_runtime)
            model_path = self._resolve_model_path(rt)
            model, class_names, predict_device = self._load_model(rt, model_path)
            probs = self._run_batches(model, class_names, predict_device)
            self.signals.progress.emit(100, "Inference complete!")
            self.signals.success.emit({"probs": probs, "class_names": class_names})

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class LogitsUMAPWorker(QRunnable):
    """Compute UMAP projection from a probability / logit matrix (N × C)."""

    def __init__(self, probs, n_neighbors: int = 15, min_dist: float = 0.1):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.probs = probs
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            n, c = self.probs.shape
            self.signals.progress.emit(0, f"Input: {n:,} images × {c} classes")

            from ..core.viz.umap_reduce import UMAPReducer

            self.signals.progress.emit(
                10,
                f"Computing UMAP in model logits space (n_neighbors={self.n_neighbors})...",
            )
            reducer = UMAPReducer(
                n_neighbors=min(self.n_neighbors, max(2, n - 1)),
                min_dist=self.min_dist,
                metric="euclidean",
            )
            coords = reducer.fit_transform(self.probs)
            self.signals.progress.emit(100, "Model-space UMAP complete!")
            self.signals.success.emit({"coords": coords})

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class LogitsPCAWorker(QRunnable):
    """Compute a 2D PCA projection from a probability / logit matrix (N × C)."""

    def __init__(self, probs):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.probs = probs
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            import numpy as np

            mat = np.asarray(self.probs, dtype=np.float64)
            if mat.ndim != 2:
                raise ValueError("Expected 2D probability/logit matrix for PCA")
            n, c = mat.shape
            self.signals.progress.emit(0, f"Input: {n:,} images × {c} classes")

            if n < 2:
                raise ValueError("Need at least 2 samples to compute PCA")

            self.signals.progress.emit(20, "Centering logits/probabilities...")
            centered = mat - mat.mean(axis=0, keepdims=True)

            self.signals.progress.emit(50, "Running SVD for PCA projection...")
            _u, _s, vt = np.linalg.svd(centered, full_matrices=False)

            components = vt[:2].T if vt.shape[0] >= 2 else vt[:1].T
            coords = centered @ components
            if coords.shape[1] == 1:
                coords = np.concatenate(
                    [coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)],
                    axis=1,
                )

            self.signals.progress.emit(100, "Model-space PCA complete!")
            self.signals.success.emit({"coords": coords.astype(np.float32)})

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class ALBatchWorker(QRunnable):
    """Select the next active learning labeling batch using acquisition strategies."""

    def __init__(
        self,
        embeddings,
        probs,
        labeled_mask,
        cluster_assignments=None,
        batch_size: int = 50,
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.embeddings = embeddings
        self.probs = probs
        self.labeled_mask = labeled_mask
        self.cluster_assignments = cluster_assignments
        self.batch_size = batch_size
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()
            import numpy as np

            from ..core.al.acquisition import BatchAcquisition, BatchConfig

            self.signals.progress.emit(0, "Computing active learning batch...")
            unlabeled_mask = ~self.labeled_mask

            if unlabeled_mask.sum() == 0:
                self.signals.success.emit(
                    {"selected_indices": np.array([], dtype=int), "breakdown": {}}
                )
                return

            cluster_densities = None
            label_coverage = {}

            if self.cluster_assignments is not None:
                self.signals.progress.emit(20, "Computing cluster densities...")
                try:
                    from ..core.al.density import compute_cluster_densities

                    cluster_densities = compute_cluster_densities(
                        self.embeddings, self.cluster_assignments
                    )
                except Exception:
                    pass

                try:
                    from ..core.train.metrics import compute_label_coverage

                    labeled_int = np.where(self.labeled_mask, 0, -1)
                    label_coverage = compute_label_coverage(
                        labeled_int, self.cluster_assignments
                    )
                except Exception:
                    pass

            self.signals.progress.emit(50, "Running batch acquisition selection...")
            cfg = BatchConfig(
                batch_size=min(self.batch_size, int(unlabeled_mask.sum()))
            )
            acq = BatchAcquisition(cfg)
            selected, breakdown = acq.select_batch(
                embeddings=self.embeddings,
                probs=self.probs,
                unlabeled_mask=unlabeled_mask,
                cluster_assignments=self.cluster_assignments,
                label_coverage=label_coverage,
                cluster_densities=cluster_densities,
            )

            self.signals.progress.emit(100, f"Selected {len(selected)} candidates!")
            self.signals.success.emit(
                {"selected_indices": selected, "breakdown": breakdown}
            )

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class TinyCNNInferenceWorker(QRunnable):
    """Run tiny CNN classification inference on all project images.

    Supports PyTorch (cpu/mps/cuda) and ONNX/TensorRT runtimes via the
    canonical ``compute_runtime`` parameter (see ``runtime.compute_runtime``).
    ONNX inference requires a ``<stem>.onnx`` sibling of the ``.pth`` model file,
    which is auto-exported by ``runner.py`` after training.

    Returns the same ``{"probs": ndarray, "class_names": list}`` contract as
    ``YoloInferenceWorker`` so callers share a unified post-inference path.
    """

    def __init__(
        self,
        model_path: Path,
        image_paths: List[Path],
        class_names: List[str],
        compute_runtime: str = "cpu",
        batch_size: int = 64,
        force_monochrome: bool = False,
        # Deprecated — use compute_runtime instead.
        device: str = "",
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.model_path = Path(model_path)
        self.image_paths = list(image_paths)
        self.class_names = list(class_names)
        # Backward-compat: if caller only passed the old `device` kwarg, honour it.
        if (not compute_runtime or compute_runtime == "cpu") and device not in (
            "",
            "cpu",
        ):
            compute_runtime = device
        self.compute_runtime = str(compute_runtime or "cpu")
        self.batch_size = batch_size
        self.force_monochrome = bool(force_monochrome)
        self.signals = TaskSignals()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _torch_device(rt: str) -> str:
        """Map a canonical runtime string to a plain PyTorch device string."""
        if rt in ("cuda", "onnx_cuda", "tensorrt"):
            return "cuda"
        if rt in ("mps", "onnx_coreml"):
            return "mps"
        if rt in ("rocm", "onnx_rocm"):
            return "cuda"  # kept for legacy configs; ROCm is no longer supported
        return "cpu"

    @staticmethod
    def _load_batch_images(batch_paths, input_w, input_h, force_monochrome=False):
        import cv2
        import numpy as np

        tensors = []
        for path in batch_paths:
            try:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is None:
                    tensors.append(np.zeros((3, input_h, input_w), dtype=np.float32))
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if force_monochrome:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                img = cv2.resize(
                    img, (input_w, input_h), interpolation=cv2.INTER_LINEAR
                )
                tensors.append(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
            except Exception:
                tensors.append(np.zeros((3, input_h, input_w), dtype=np.float32))
        if tensors:
            return np.stack(tensors, axis=0)
        return np.zeros((0, 3, input_h, input_w), dtype=np.float32)

    def _resolve_tiny_runtime(self, rt: str) -> tuple[bool, Path]:
        onnx_path = self.model_path.with_suffix(".onnx")
        use_onnx = rt.startswith("onnx_") or rt == "tensorrt"
        if use_onnx and not onnx_path.exists():
            use_onnx = False
        return use_onnx, onnx_path

    def _run_tiny_onnx(self, rt: str, onnx_path: Path, num_images: int):
        import torch

        from ...training.tiny_model import load_tiny_onnx, run_tiny_onnx

        resolved_names = list(self.class_names)
        try:
            ckpt = torch.load(
                str(self.model_path), map_location="cpu", weights_only=False
            )
            input_w, input_h = ckpt.get("input_size", [128, 64])
            ckpt_names = ckpt.get("class_names")
            if ckpt_names:
                resolved_names = list(ckpt_names)
            force_monochrome = bool(ckpt.get("monochrome", self.force_monochrome))
        except Exception:
            input_w, input_h = 128, 64
            force_monochrome = self.force_monochrome

        session = load_tiny_onnx(str(onnx_path), compute_runtime=rt)
        self.signals.progress.emit(
            5, f"Tiny CNN ONNX inference on {num_images:,} images..."
        )
        all_probs = []
        for batch_start in range(0, num_images, self.batch_size):
            batch_paths = self.image_paths[batch_start : batch_start + self.batch_size]
            batch_np = self._load_batch_images(
                batch_paths,
                input_w,
                input_h,
                force_monochrome=force_monochrome,
            )
            if batch_np.shape[0] == 0:
                continue
            all_probs.append(run_tiny_onnx(session, batch_np))
            done = batch_start + len(batch_paths)
            pct = min(95, 5 + int(90 * done / num_images))
            self.signals.progress.emit(pct, f"Processed {done:,}/{num_images:,}")
        return all_probs, resolved_names

    def _run_tiny_torch(self, rt: str, num_images: int):
        import torch
        import torch.nn.functional as F

        from ...training.tiny_model import load_tiny_classifier

        resolved_names = list(self.class_names)
        torch_device = self._torch_device(rt)
        model, ckpt = load_tiny_classifier(str(self.model_path), device=torch_device)
        input_w, input_h = ckpt.get("input_size", [128, 64])
        ckpt_names = ckpt.get("class_names")
        if ckpt_names:
            resolved_names = list(ckpt_names)
        force_monochrome = bool(ckpt.get("monochrome", self.force_monochrome))

        self.signals.progress.emit(5, f"Tiny CNN inference on {num_images:,} images...")
        all_probs = []
        for batch_start in range(0, num_images, self.batch_size):
            batch_paths = self.image_paths[batch_start : batch_start + self.batch_size]
            batch_np = self._load_batch_images(
                batch_paths,
                input_w,
                input_h,
                force_monochrome=force_monochrome,
            )
            if batch_np.shape[0] == 0:
                continue
            x = torch.from_numpy(batch_np).to(torch_device)
            with torch.no_grad():
                logits = model(x)
                all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            done = batch_start + len(batch_paths)
            pct = min(95, 5 + int(90 * done / num_images))
            self.signals.progress.emit(pct, f"Processed {done:,}/{num_images:,}")
        return all_probs, resolved_names

    @staticmethod
    def _finalize_tiny_probs(all_probs, resolved_names, num_images):
        import numpy as np

        if all_probs:
            return np.concatenate(all_probs, axis=0)
        n_cls = max(len(resolved_names), 1)
        return np.full((num_images, n_cls), 1.0 / n_cls)

    @Slot()
    def run(self):
        try:
            self.signals.started.emit()

            from ...runtime.compute_runtime import _normalize_runtime

            rt = _normalize_runtime(self.compute_runtime)
            use_onnx, onnx_path = self._resolve_tiny_runtime(rt)

            self.signals.progress.emit(
                0, f"Loading tiny CNN ({rt}): {self.model_path.name}..."
            )

            num_images = len(self.image_paths)
            if use_onnx:
                all_probs, resolved_names = self._run_tiny_onnx(
                    rt, onnx_path, num_images
                )
            else:
                all_probs, resolved_names = self._run_tiny_torch(rt, num_images)

            result_probs = self._finalize_tiny_probs(
                all_probs, resolved_names, num_images
            )

            self.signals.progress.emit(100, "Tiny CNN inference complete!")
            self.signals.success.emit(
                {"probs": result_probs, "class_names": resolved_names}
            )

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class TorchvisionInferenceWorker(QRunnable):
    """Run torchvision Custom CNN classification inference on all project images.

    Supports PyTorch (cpu/mps/cuda) and ONNX runtimes via
    ``compute_runtime``. Output contract: same as TinyCNNInferenceWorker —
    emits ``{"probs": ndarray(N, C), "class_names": list}`` via success signal.
    """

    def __init__(
        self,
        model_path: Path,
        image_paths: List[Path],
        class_names: List[str],
        input_size: int = 224,
        compute_runtime: str = "cpu",
        batch_size: int = 64,
        force_monochrome: bool = False,
    ):
        super().__init__()
        self.setAutoDelete(False)
        self.model_path = Path(model_path)
        self.image_paths = list(image_paths)
        self.class_names = list(class_names)
        self.input_size = input_size
        self.compute_runtime = str(compute_runtime or "cpu")
        self.batch_size = batch_size
        self.force_monochrome = bool(force_monochrome)
        self.signals = TaskSignals()

    @staticmethod
    def _torch_device(rt: str) -> str:
        """Map canonical runtime to PyTorch device string."""
        if rt in ("cuda", "onnx_cuda", "tensorrt"):
            return "cuda"
        if rt in ("mps", "onnx_coreml"):
            return "mps"
        if rt in ("rocm", "onnx_rocm"):
            return "cuda"  # kept for legacy configs; ROCm is no longer supported
        return "cpu"

    def _build_transform(self):
        from torchvision import transforms

        from ...training.torchvision_model import get_classifier_normalization_stats

        sz = self.input_size
        mean, std = get_classifier_normalization_stats(monochrome=self.force_monochrome)
        transform_steps = [transforms.Resize((sz, sz))]
        if self.force_monochrome:
            transform_steps.append(transforms.Grayscale(num_output_channels=3))
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return transforms.Compose(transform_steps)

    def _create_infer_fn(self, rt: str):
        import torch

        use_onnx = rt.startswith("onnx_") or rt == "tensorrt"
        onnx_path = self.model_path.with_suffix(".onnx")
        if use_onnx and onnx_path.exists():
            import onnxruntime as ort

            from ...runtime.compute_runtime import derive_onnx_execution_providers

            providers = derive_onnx_execution_providers(rt)
            sess = ort.InferenceSession(str(onnx_path), providers=providers)
            input_name = sess.get_inputs()[0].name

            def _infer(batch_np):
                return sess.run(None, {input_name: batch_np})[0]

            return _infer

        from ...training.torchvision_model import load_torchvision_classifier

        device = self._torch_device(rt)
        model, _ = load_torchvision_classifier(str(self.model_path), device=device)

        def _infer(batch_np):
            t = torch.from_numpy(batch_np).to(device)
            with torch.no_grad():
                return model(t).cpu().numpy()

        return _infer

    def _load_batch_tensors(self, batch_paths, transform):
        import numpy as np
        from PIL import Image

        batch_tensors = []
        sz = self.input_size
        for path in batch_paths:
            try:
                img = Image.open(str(path)).convert("RGB")
                if self.force_monochrome:
                    img = img.convert("L").convert("RGB")
                batch_tensors.append(transform(img).numpy())
            except Exception:
                batch_tensors.append(np.zeros((3, sz, sz), dtype=np.float32))
        return np.stack(batch_tensors).astype(np.float32)

    def _resolve_force_monochrome(self) -> bool:
        try:
            import torch

            ckpt = torch.load(
                str(self.model_path), map_location="cpu", weights_only=False
            )
            if isinstance(ckpt, dict):
                return bool(ckpt.get("monochrome", self.force_monochrome))
        except Exception:
            pass
        return self.force_monochrome

    @staticmethod
    def _softmax_numpy(logits):
        import numpy as np

        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def _run_inference_batches(self, infer_fn, transform):
        import numpy as np

        all_probs = []
        total = len(self.image_paths)
        for i in range(0, total, self.batch_size):
            batch_paths = self.image_paths[i : i + self.batch_size]
            batch_np = self._load_batch_tensors(batch_paths, transform)
            logits = infer_fn(batch_np)
            all_probs.append(self._softmax_numpy(logits))
            done = min(i + self.batch_size, total)
            pct = int(done * 100 / total) if total else 100
            self.signals.progress.emit(pct, f"Inferring {done}/{total}")

        if all_probs:
            return np.concatenate(all_probs, axis=0)
        return np.zeros((0, len(self.class_names)))

    @Slot()
    def run(self) -> None:
        try:
            self.signals.started.emit()
            from ...runtime.compute_runtime import _normalize_runtime

            rt = _normalize_runtime(self.compute_runtime)
            self.force_monochrome = self._resolve_force_monochrome()
            transform = self._build_transform()
            infer_fn = self._create_infer_fn(rt)
            all_probs_np = self._run_inference_batches(infer_fn, transform)
            self.signals.success.emit(
                {"probs": all_probs_np, "class_names": self.class_names}
            )

        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class AprilTagAutoLabelWorker(QRunnable):
    """Background worker that runs AprilTag auto-labeling on a list of images.

    The caller (mainwindow) is responsible for:
    - pre-filtering image_paths to unlabeled images only
    - writing the labeling scheme before starting the worker
    - connecting signals to update the UI

    This worker uses ``db`` only for writing labels — it never reads from it.
    """

    def __init__(
        self,
        image_paths: List[Path],
        config,  # AprilTagConfig — imported lazily to avoid hard dep at module load
        threshold: float,
        db,  # ClassKitDB — imported lazily
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.image_paths = image_paths
        self.config = config
        self.threshold = threshold
        self.db = db
        self.signals = TaskSignals()
        self._canceled = False

    def cancel(self) -> None:
        self._canceled = True

    @Slot()
    def run(self) -> None:
        from ..core.autolabel.apriltag import autolabel_images

        try:
            self.signals.started.emit()
            total = len(self.image_paths)
            if total == 0:
                self.signals.success.emit(
                    {"n_labeled": 0, "n_skipped": 0, "n_no_tag": 0}
                )
                return

            self.signals.progress.emit(0, f"Auto-labeling {total:,} images...")

            n_labeled = 0
            n_skipped = 0
            n_no_tag = 0
            batch_size = 100

            for batch_start in range(0, total, batch_size):
                if self._canceled:
                    self.signals.error.emit("Canceled by user")
                    return

                batch = self.image_paths[batch_start : batch_start + batch_size]
                results = autolabel_images(batch, self.config, self.threshold)

                updates: Dict[str, tuple] = {}
                for path, result in zip(batch, results):
                    if result.label is not None:
                        updates[str(path)] = (result.label, result.confidence)
                        if result.label == "no_tag":
                            n_no_tag += 1
                        else:
                            n_labeled += 1
                    else:
                        n_skipped += 1

                if updates:
                    self.db.update_labels_with_confidence_batch(
                        updates,
                        label_source="auto_apriltag",
                        verified=False,
                    )

                done = min(batch_start + batch_size, total)
                pct = int(done * 100 / total)
                self.signals.progress.emit(
                    pct,
                    f"Labeled {n_labeled + n_no_tag}, skipped {n_skipped} of {total}",
                )

            self.signals.success.emit(
                {"n_labeled": n_labeled, "n_skipped": n_skipped, "n_no_tag": n_no_tag}
            )

        except Exception as exc:
            traceback.print_exc()
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()
            self.signals.finished.emit()
