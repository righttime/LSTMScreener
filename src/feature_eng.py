from __future__ import annotations
"""Feature engineering: technical indicators + sequence builder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional

from src import config

class FeatureEngineer:
    """Adds technical indicators and creates LSTM-ready sequences."""

    def __init__(self, seq_len: int = 60):
        self.seq_len = seq_len
        self.scaler: Optional[MinMaxScaler] = None
        self._feature_columns: list = []

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to OHLCV DataFrame.

        Adds: MA5/20/60/120, RSI, MACD, Bollinger Bands,
              volume ratio, price momentum, etc.
        """
        df = df.copy()
        close = df["close"]

        # ---- Moving Averages ----
        for w in config.MA_WINDOWS:
            df[f"ma{w}"] = close.rolling(w).mean()

        # ---- RSI ----
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(config.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(config.RSI_PERIOD).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # ---- MACD ----
        ema_fast = close.ewm(span=config.MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=config.MACD_SLOW, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ---- Bollinger Bands ----
        bb_mid = close.rolling(config.BB_PERIOD).mean()
        bb_std = close.rolling(config.BB_PERIOD).std()
        df["bb_upper"] = bb_mid + config.BB_STD * bb_std
        df["bb_lower"] = bb_mid - config.BB_STD * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid

        # ---- Volume ratio (vs 20-day avg) ----
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)

        # ---- Price momentum ----
        df["momentum_5"] = close / close.shift(5) - 1
        df["momentum_20"] = close / close.shift(20) - 1

        # ---- Price position within BB ----
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

        # ---- Daily return ----
        df["return_1d"] = close.pct_change()

        # ---- Volatility (20-day) ----
        df["volatility_20"] = df["return_1d"].rolling(20).std()

        return df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order feature columns for the model."""
        base = ["open", "high", "low", "close", "volume"]
        feat = [
            "ma5", "ma20", "ma60", "ma120",
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_lower", "bb_width", "bb_position",
            "vol_ratio", "momentum_5", "momentum_20",
            "return_1d", "volatility_20",
        ]
        cols = [c for c in base + feat if c in df.columns]
        self._feature_columns = cols
        return df[cols]

    def create_sequences(self, df: pd.DataFrame,
                         seq_len: int = None, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create (X, y) sequences.

        X : (samples, seq_len, features)
        y : (samples,) — 1 if next-day return > 0 else 0
        """
        if seq_len is None:
            seq_len = self.seq_len

        arr = df.values
        X, y = [], []
        for i in range(len(arr) - seq_len - horizon):
            seq = arr[i : i + seq_len]
            price_now = arr[i + seq_len - 1, 3]   # close price column
            price_next = arr[i + seq_len, 3]
            ret = 1.0 if price_next >= price_now else 0.0
            X.append(seq)
            y.append(ret)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def normalize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize features using MinMaxScaler.

        Parameters
        ----------
        X   : feature array (n_samples, n_features)
        fit : if True, fit the scaler; if False, transform only
        """
        if fit:
            self.scaler = MinMaxScaler()
            self.scaler.fit(X)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call normalize(fit=True) first.")
        return self.scaler.transform(X)

    def normalize_train(self, df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
        """Fit scaler on df and return scaled array + scaler."""
        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(df.values)
        return scaled, self.scaler