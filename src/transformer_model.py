"""
Phase 2: Advanced AI - Transformer Encoder for Gesture Sequence Classification
Processes sequences of 30-60 frames of hand landmarks to classify dynamic gestures.
Transformers handle long-range temporal dependencies more efficiently than LSTMs
and are the current state-of-the-art for sequence analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import csv
import json
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Dataset
# ============================================================================

class GestureSequenceDataset(Dataset):
    """PyTorch Dataset for gesture landmark sequences stored as CSV files."""

    GESTURE_CLASSES = [
        'forward', 'backward', 'turn_left', 'turn_right',
        'stop', 'slow', 'fast', 'idle'
    ]

    def __init__(self, data_dir: str, sequence_length: int = 45,
                 num_landmarks: int = 21, coords_per_landmark: int = 3):
        """
        Args:
            data_dir: Root directory containing gesture subdirectories.
            sequence_length: Expected number of frames per sequence.
            num_landmarks: Number of hand landmarks (21 for MediaPipe).
            coords_per_landmark: Coordinates per landmark (3 for X,Y,Z).
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.num_landmarks = num_landmarks
        self.coords_per_landmark = coords_per_landmark
        self.feature_dim = num_landmarks * coords_per_landmark  # 63

        self.label_to_idx = {label: idx for idx, label in enumerate(self.GESTURE_CLASSES)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        self.samples = []  # List of (file_path, label_idx)
        self._load_file_list()

    def _load_file_list(self):
        """Scan data directory and build list of (filepath, label) pairs."""
        for gesture in self.GESTURE_CLASSES:
            gesture_dir = os.path.join(self.data_dir, gesture)
            if not os.path.exists(gesture_dir):
                continue
            for filename in sorted(os.listdir(gesture_dir)):
                if filename.endswith('.csv'):
                    filepath = os.path.join(gesture_dir, filename)
                    label_idx = self.label_to_idx[gesture]
                    self.samples.append((filepath, label_idx))

        print(f"Loaded {len(self.samples)} samples from {self.data_dir}")

    def _load_csv_sequence(self, filepath: str) -> np.ndarray:
        """
        Load a CSV file and return the landmark sequence as a numpy array.

        Returns:
            Array of shape (sequence_length, feature_dim)
        """
        frames = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                # Skip frame_idx (first column), take landmark values
                values = [float(v) for v in row[1:]]
                frames.append(values)

        sequence = np.array(frames, dtype=np.float32)

        # Pad or truncate to sequence_length
        if len(sequence) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(sequence), self.feature_dim),
                               dtype=np.float32)
            sequence = np.vstack([sequence, padding])
        elif len(sequence) > self.sequence_length:
            sequence = sequence[:self.sequence_length]

        return sequence

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label_idx = self.samples[idx]
        sequence = self._load_csv_sequence(filepath)
        return torch.FloatTensor(sequence), torch.LongTensor([label_idx]).squeeze()


# ============================================================================
# Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequence ordering."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1]) if div_term.shape[0] > d_model // 2 else torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input. x shape: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# Transformer Encoder Model
# ============================================================================

class GestureTransformer(nn.Module):
    """
    Transformer Encoder for gesture sequence classification.

    Architecture:
    1. Linear projection of 63-dim landmark features to d_model dimensions
    2. Positional encoding for temporal ordering
    3. N Transformer Encoder layers with multi-head self-attention
    4. Global average pooling over the sequence dimension
    5. Classification head with dropout
    """

    def __init__(self, input_dim: int = 63, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 256, num_classes: int = 8,
                 max_seq_len: int = 60, dropout: float = 0.1):
        """
        Args:
            input_dim: Input feature dimension (21 landmarks * 3 coords = 63).
            d_model: Transformer model dimension.
            nhead: Number of attention heads.
            num_layers: Number of Transformer encoder layers.
            dim_feedforward: Feedforward network dimension.
            num_classes: Number of gesture classes.
            max_seq_len: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Project input features to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x, mask=mask)  # (batch, seq_len, d_model)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        logits = self.classifier(x)  # (batch, num_classes)

        return logits

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict gesture class with confidence scores.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Tuple of (predicted_class_indices, confidence_scores)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence, predicted = torch.max(probs, dim=-1)
        return predicted, confidence


# ============================================================================
# Performance Metrics
# ============================================================================

class PerformanceMetrics:
    """
    PG-specific performance metrics for real-time gesture recognition:
    1. Processing Time: Must be < 300ms for real-time validity
    2. Prediction Vector Validity: No flickering/spurious labels in sequence
    """

    def __init__(self, stability_window: int = 5, confidence_threshold: float = 0.6):
        """
        Args:
            stability_window: Number of consecutive predictions to check for stability.
            confidence_threshold: Minimum confidence to accept a prediction.
        """
        self.stability_window = stability_window
        self.confidence_threshold = confidence_threshold
        self.prediction_history: List[int] = []
        self.timing_history: List[float] = []

    def measure_inference_time(self, model: GestureTransformer,
                                sample_input: torch.Tensor,
                                num_runs: int = 100) -> Dict[str, float]:
        """
        Measure model inference time.

        Args:
            model: The gesture transformer model.
            sample_input: Sample input tensor.
            num_runs: Number of inference runs for averaging.

        Returns:
            Dictionary with timing statistics in milliseconds.
        """
        model.eval()
        times = []

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)

        # Measure
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(sample_input)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        times = np.array(times)
        results = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'meets_realtime': bool(np.percentile(times, 95) < 300)
        }

        print(f"\nInference Time Metrics:")
        print(f"  Mean: {results['mean_ms']:.2f} ms")
        print(f"  Std:  {results['std_ms']:.2f} ms")
        print(f"  P95:  {results['p95_ms']:.2f} ms")
        print(f"  P99:  {results['p99_ms']:.2f} ms")
        print(f"  Meets <300ms requirement: {'YES' if results['meets_realtime'] else 'NO'}")

        return results

    def check_prediction_stability(self, prediction: int, confidence: float) -> Dict:
        """
        Check if predictions are stable (no flickering).

        Args:
            prediction: Current predicted class index.
            confidence: Current prediction confidence.

        Returns:
            Dictionary with stability metrics.
        """
        self.prediction_history.append(prediction)

        # Keep only recent history
        if len(self.prediction_history) > self.stability_window * 3:
            self.prediction_history = self.prediction_history[-self.stability_window * 3:]

        result = {
            'current_prediction': prediction,
            'confidence': confidence,
            'is_confident': confidence >= self.confidence_threshold,
            'is_stable': False,
            'stable_label': None,
            'flicker_count': 0
        }

        if len(self.prediction_history) >= self.stability_window:
            recent = self.prediction_history[-self.stability_window:]
            # Count unique predictions in window
            unique_preds = len(set(recent))
            result['flicker_count'] = unique_preds - 1

            # Stable if all predictions in window are the same
            if unique_preds == 1 and confidence >= self.confidence_threshold:
                result['is_stable'] = True
                result['stable_label'] = prediction

        return result

    def get_summary(self) -> Dict:
        """Get overall performance summary."""
        if not self.prediction_history:
            return {'total_predictions': 0}

        total = len(self.prediction_history)
        changes = sum(1 for i in range(1, total)
                      if self.prediction_history[i] != self.prediction_history[i-1])

        return {
            'total_predictions': total,
            'label_changes': changes,
            'stability_ratio': 1.0 - (changes / max(total - 1, 1)),
            'unique_labels_seen': len(set(self.prediction_history))
        }


# ============================================================================
# Training Utilities
# ============================================================================

def train_model(model: GestureTransformer, train_loader: DataLoader,
                val_loader: Optional[DataLoader] = None,
                epochs: int = 50, lr: float = 1e-3,
                device: str = 'cpu', save_path: str = '../models/gesture_transformer.pth'
                ) -> Dict[str, List[float]]:
    """
    Train the Gesture Transformer model.

    Args:
        model: GestureTransformer model.
        train_loader: Training data loader.
        val_loader: Optional validation data loader.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Device to train on ('cpu', 'cuda', 'mps').
        save_path: Path to save the best model.

    Returns:
        Dictionary with training history.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        scheduler.step()

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)

                    val_loss += loss.item() * batch_x.size(0)
                    _, predicted = torch.max(logits, 1)
                    val_correct += (predicted == batch_y).sum().item()
                    val_total += batch_y.size(0)

            avg_val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': avg_val_loss
                }, save_path)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")
        else:
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f}")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    return history


def load_model(model_path: str, device: str = 'cpu', **model_kwargs) -> GestureTransformer:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to the saved model checkpoint.
        device: Device to load model onto.
        **model_kwargs: Model architecture parameters.

    Returns:
        Loaded GestureTransformer model.
    """
    model = GestureTransformer(**model_kwargs)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path} (Val Acc: {checkpoint.get('val_acc', 'N/A')})")
    return model


# ============================================================================
# Training & inference wrappers (pipeline integration)
# ============================================================================

GestureTransformerEncoder = GestureTransformer


class GesturePredictor:
    """Load a trained transformer and classify landmark sequences."""

    def __init__(
        self,
        model_path: str,
        gesture_labels: Optional[List[str]] = None,
        sequence_length: int = 45,
        device: str = "cpu",
    ):
        self.sequence_length = sequence_length
        self.device = device
        self.gesture_labels = gesture_labels or GestureSequenceDataset.GESTURE_CLASSES
        self.label_to_idx = {l: i for i, l in enumerate(self.gesture_labels)}
        self.idx_to_label = {i: l for l, i in self.label_to_idx.items()}

        self.model = GestureTransformer(
            input_dim=63,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=256,
            num_classes=len(self.gesture_labels),
            max_seq_len=max(sequence_length, 60),
            dropout=0.1,
        )
        if os.path.isfile(model_path):
            self.model = load_model(
                model_path,
                device=device,
                input_dim=63,
                d_model=128,
                nhead=8,
                num_layers=4,
                dim_feedforward=256,
                num_classes=len(self.gesture_labels),
                max_seq_len=max(sequence_length, 60),
                dropout=0.1,
            )
        self.model.eval()

    def _prepare_sequence(self, sequence: np.ndarray) -> torch.Tensor:
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[-1] == 63:
            features = arr
        else:
            flat = []
            for frame in arr:
                frame = np.asarray(frame, dtype=np.float32)
                flat.append(frame.reshape(-1))
            features = np.stack(flat, axis=0)

        if len(features) < self.sequence_length:
            pad = np.zeros(
                (self.sequence_length - len(features), features.shape[1]),
                dtype=np.float32,
            )
            features = np.vstack([features, pad])
        elif len(features) > self.sequence_length:
            features = features[-self.sequence_length :]

        return torch.from_numpy(features).unsqueeze(0)

    def predict(self, sequence: np.ndarray) -> Dict:
        tensor = self._prepare_sequence(sequence).to(self.device)
        with torch.no_grad():
            predicted, confidence = self.model.predict(tensor)
        idx = int(predicted[0].item())
        conf = float(confidence[0].item())
        gesture = self.idx_to_label.get(idx, "idle")
        return {"gesture": gesture, "confidence": conf, "class_index": idx}


class GestureModelTrainer:
    """Train GestureTransformer on CSV sequences under data_dir/<gesture>/."""

    def __init__(
        self,
        num_classes: int = 8,
        sequence_length: int = 45,
        num_landmarks: int = 21,
        landmark_dims: int = 3,
        data_dir: Optional[str] = None,
    ):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.data_dir = data_dir

    def train(
        self,
        data_path: Optional[str] = None,
        epochs: int = 50,
        batch_size: int = 32,
        save_path: str = "../models/gesture_transformer.pth",
        lr: float = 1e-3,
    ) -> Dict:
        root = self.data_dir
        if root is None and data_path:
            root = os.path.dirname(os.path.abspath(data_path))
        if root is None:
            root = os.path.join(os.path.dirname(__file__), "..", "data")

        dataset = GestureSequenceDataset(
            data_dir=root,
            sequence_length=self.sequence_length,
        )
        if len(dataset) == 0:
            raise FileNotFoundError(
                f"No CSV samples found under {root}. "
                "Run: python src/data_collector.py and record gestures, or press 's' for synthetic data."
            )

        n_val = max(1, int(0.2 * len(dataset)))
        n_train = len(dataset) - n_val
        train_set, val_set = torch.utils.data.random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        model = GestureTransformer(
            input_dim=63,
            num_classes=self.num_classes,
            max_seq_len=self.sequence_length,
        )
        return train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            save_path=save_path,
            lr=lr,
        )


# ============================================================================
# Quick Test / Demo
# ============================================================================

def quick_test():
    """Quick test to verify model architecture and inference speed."""
    print("=" * 60)
    print("  Gesture Transformer - Quick Architecture Test")
    print("=" * 60)

    # Create model
    model = GestureTransformer(
        input_dim=63,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        num_classes=8,
        max_seq_len=60,
        dropout=0.1
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 45
    input_dim = 63
    sample_input = torch.randn(batch_size, seq_len, input_dim)

    print(f"\nInput shape: {sample_input.shape}")
    output = model(sample_input)
    print(f"Output shape: {output.shape}")

    # Test prediction
    predicted, confidence = model.predict(sample_input)
    print(f"Predicted classes: {predicted.tolist()}")
    print(f"Confidence scores: {[f'{c:.3f}' for c in confidence.tolist()]}")

    # Measure inference time
    metrics = PerformanceMetrics()
    single_input = torch.randn(1, seq_len, input_dim)
    timing = metrics.measure_inference_time(model, single_input, num_runs=100)

    print("\nQuick test PASSED!")
    return model


if __name__ == '__main__':
    quick_test()
