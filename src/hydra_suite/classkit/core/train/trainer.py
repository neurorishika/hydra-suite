"""
Trainer for embedding-head classifiers.

Fast training on frozen embeddings with calibration.
"""

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    optim = None
    TensorDataset = None
    DataLoader = None
    np = None

from pathlib import Path
from typing import Any, Dict, Optional

from .calibrate import TemperatureScaling
from .head_models import create_head
from .metrics import EvalMetrics, compute_metrics


class EmbeddingHeadTrainer:
    """
    Train lightweight classifier on frozen embeddings.
    """

    def __init__(
        self,
        model_type: str = "linear",
        input_dim: int = 768,
        num_classes: int = 10,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        """
        Args:
            model_type: 'linear' or 'mlp'
            input_dim: Embedding dimension
            num_classes: Number of classes
            hidden_dim: Hidden dimension for MLP
            dropout: Dropout rate
            device: 'cpu', 'cuda', or 'mps'
        """
        if torch is None:
            raise ImportError("torch required for training")

        self.model_type = model_type
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device

        # Create model
        self.model = create_head(
            model_type=model_type,
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)

        self.calibrator = TemperatureScaling()
        self.is_calibrated = False

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        val_embeddings: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        batch_size: int = 256,
        epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        early_stop_patience: int = 10,
        ignore_label_index: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the classifier.

        Args:
            train_embeddings: (N, D) training embeddings
            train_labels: (N,) training labels
            val_embeddings: (M, D) validation embeddings (optional)
            val_labels: (M,) validation labels (optional)
            batch_size: Batch size
            epochs: Maximum epochs
            lr: Learning rate
            weight_decay: L2 regularization
            early_stop_patience: Early stopping patience
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_embeddings), torch.LongTensor(train_labels)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        has_val = val_embeddings is not None and val_labels is not None

        # Optimizer and loss
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion_kwargs = {}
        if ignore_label_index is not None:
            criterion_kwargs["ignore_index"] = int(ignore_label_index)
        criterion = nn.CrossEntropyLoss(**criterion_kwargs)

        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            train_count = 0

            for embs, labels in train_loader:
                embs = embs.to(self.device)
                labels = labels.to(self.device)
                if ignore_label_index is not None:
                    active_count = int((labels != int(ignore_label_index)).sum().item())
                    if active_count <= 0:
                        continue
                else:
                    active_count = len(labels)

                optimizer.zero_grad()
                logits = self.model(embs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * active_count
                train_count += active_count

            train_loss /= max(1, train_count)
            history["train_loss"].append(train_loss)

            # Validate
            if has_val:
                val_loss, val_acc = self._validate(
                    val_embeddings,
                    val_labels,
                    criterion,
                    ignore_label_index=ignore_label_index,
                )
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"train_loss: {train_loss:.4f}, "
                        f"val_loss: {val_loss:.4f}, "
                        f"val_acc: {val_acc:.4f}"
                    )

                # Early stopping with best-state checkpointing
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stop_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f}")

        # Restore best model weights when early stopping was active
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return history

    def _validate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        criterion: nn.Module,
        ignore_label_index: Optional[int] = None,
    ) -> tuple:
        """Compute validation loss and accuracy."""
        self.model.eval()

        with torch.no_grad():
            embs_tensor = torch.FloatTensor(embeddings).to(self.device)
            labels_tensor = torch.LongTensor(labels).to(self.device)

            logits = self.model(embs_tensor)
            loss = criterion(logits, labels_tensor).item()

            preds = logits.argmax(dim=1)
            if ignore_label_index is not None:
                mask = labels_tensor != int(ignore_label_index)
                if bool(mask.any().item()):
                    preds = preds[mask]
                    labels_tensor = labels_tensor[mask]
                    acc = (preds == labels_tensor).float().mean().item()
                else:
                    acc = 0.0
            else:
                acc = (preds == labels_tensor).float().mean().item()

        return loss, acc

    def calibrate(self, val_embeddings: np.ndarray, val_labels: np.ndarray):
        """
        Calibrate model on validation set using temperature scaling.

        Args:
            val_embeddings: (N, D) validation embeddings
            val_labels: (N,) validation labels
        """
        # Get logits
        logits = self.predict_logits(val_embeddings)

        # Fit temperature
        self.calibrator.fit(logits, val_labels, device=self.device)
        self.is_calibrated = True

    def predict_logits(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict raw logits.

        Args:
            embeddings: (N, D) embeddings

        Returns:
            logits: (N, num_classes) raw logits
        """
        self.model.eval()

        with torch.no_grad():
            embs_tensor = torch.FloatTensor(embeddings).to(self.device)
            logits = self.model(embs_tensor)

        return logits.cpu().numpy()

    def predict_proba(
        self, embeddings: np.ndarray, calibrated: bool = True
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            embeddings: (N, D) embeddings
            calibrated: Use calibrated probabilities if available

        Returns:
            probs: (N, num_classes) probabilities
        """
        logits = self.predict_logits(embeddings)

        if calibrated and self.is_calibrated:
            # Apply temperature scaling
            probs = self.calibrator.transform(logits)
        else:
            # Uncalibrated softmax
            probs = torch.softmax(torch.FloatTensor(logits), dim=1).numpy()

        return probs

    def predict(self, embeddings: np.ndarray, calibrated: bool = True) -> np.ndarray:
        """
        Predict class labels.

        Args:
            embeddings: (N, D) embeddings
            calibrated: Use calibrated probabilities

        Returns:
            predictions: (N,) predicted labels
        """
        probs = self.predict_proba(embeddings, calibrated=calibrated)
        return probs.argmax(axis=1)

    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        calibrated: bool = True,
        class_names: Optional[list] = None,
    ) -> EvalMetrics:
        """
        Evaluate model on test set.

        Args:
            embeddings: (N, D) embeddings
            labels: (N,) ground-truth labels
            calibrated: Use calibrated probabilities
            class_names: Optional class names

        Returns:
            EvalMetrics object
        """
        predictions = self.predict(embeddings, calibrated=calibrated)
        return compute_metrics(predictions, labels, class_names=class_names)

    def save(self, path: Path):
        """Save model and calibrator."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_type": self.model_type,
                "input_dim": self.input_dim,
                "num_classes": self.num_classes,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "temperature": self.calibrator.temperature,
                "is_calibrated": self.is_calibrated,
            },
            path,
        )

    def load(self, path: Path):
        """Load model and calibrator."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Recreate model
        self.model = create_head(
            model_type=checkpoint["model_type"],
            input_dim=checkpoint["input_dim"],
            num_classes=checkpoint["num_classes"],
            hidden_dim=checkpoint.get("hidden_dim"),
            dropout=checkpoint.get("dropout", 0.1),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state"])

        # Load calibration
        self.calibrator.temperature = checkpoint.get("temperature", 1.0)
        self.is_calibrated = checkpoint.get("is_calibrated", False)
