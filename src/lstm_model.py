from __future__ import annotations
"""LSTM + Attention model (PyTorch)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from typing import Tuple

from src import config

class LSTMAttention(nn.Module):
    """LSTM with Bahdanau Attention for sequence prediction."""

    def __init__(self, input_size: int, hidden_size: int = None,
                 num_layers: int = None, dropout: float = None):
        hidden_size = hidden_size or config.HIDDEN_SIZE
        num_layers  = num_layers  or config.NUM_LAYERS
        dropout     = dropout     or config.DROPOUT

        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0,
        )

        self.attn_weights = nn.Linear(hidden_size, 1, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attn_weights(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        out = self.fc(context)
        return out.squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class LSTMModel:
    """Wrapper class for training, prediction, and persistence of LSTMAttention."""

    def __init__(self, input_size: int = 17, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model: LSTMAttention = LSTMAttention(
            input_size  = input_size,
            hidden_size = config.HIDDEN_SIZE,
            num_layers  = config.NUM_LAYERS,
            dropout     = config.DROPOUT,
        ).to(self.device)
        self._trained = False

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor,
              X_val: torch.Tensor = None, y_val: torch.Tensor = None,
              epochs: int = None, batch_size: int = None,
              early_stop: int = None) -> dict:
        """Train the model and return training history."""
        from sklearn.metrics import accuracy_score

        epochs     = epochs     or config.EPOCHS
        batch_size = batch_size or config.BATCH_SIZE
        early_stop = early_stop or config.EARLY_STOP_PATIENCE

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        if X_val is not None:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            train_loss = sum(epoch_losses) / len(epoch_losses)
            history["train_loss"].append(train_loss)

            # Accuracy
            self.model.eval()
            with torch.no_grad():
                preds  = (torch.sigmoid(self.model(X_train)) > 0.5).cpu().numpy()
                acc    = accuracy_score(y_train.cpu().numpy(), preds)
                history["train_acc"].append(acc)

            if X_val is not None and y_val is not None:
                with torch.no_grad():
                    val_out  = self.model(X_val)
                    val_loss = criterion(val_out, y_val).item()
                    val_preds = (torch.sigmoid(val_out) > 0.5).cpu().numpy()
                    val_acc   = accuracy_score(y_val.cpu().numpy(), val_preds)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stop:
                    break

        self._trained = True
        return history

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Return binary predictions (0/1) for input sequences."""
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X)).cpu().numpy()
        return (probs > 0.5).astype(int)

    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        """Return probability of price going up."""
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X)).cpu().numpy()
        return probs

    def save(self, path: str):
        """Save model state dict to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Load model state dict from path."""
        self.model.load_state_dict(torch.load(
            Path(path), map_location=self.device, weights_only=True
        ))
        self._trained = True

    @property
    def is_trained(self) -> bool:
        return self._trained


# Module-level helper (backward compatibility)
def build_model(input_size: int, device: str = "cpu") -> LSTMAttention:
    model = LSTMAttention(
        input_size  = input_size,
        hidden_size = config.HIDDEN_SIZE,
        num_layers  = config.NUM_LAYERS,
        dropout     = config.DROPOUT,
    ).to(device)
    return model