"""Tests for LSTMScreener modules (OOP refactored)."""
import numpy as np
import pandas as pd
import torch
from src.feature_eng import FeatureEngineer
from src.lstm_model import LSTMAttention, LSTMModel
from src import backtest

def test_feature_engineering():
    """Smoke-test feature engineering pipeline (OOP)."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    df = pd.DataFrame({
        "date":   dates,
        "open":   close * 0.99,
        "high":   close * 1.02,
        "low":    close * 0.97,
        "close":  close,
        "volume": np.random.randint(1e6, 5e6, 100),
    })

    engineer = FeatureEngineer(seq_len=60)
    df = engineer.add_indicators(df)
    assert "ma5"   in df.columns
    assert "rsi"   in df.columns
    assert "macd"  in df.columns
    assert "bb_upper" in df.columns

    feat = engineer.build_features(df)
    assert feat.shape[1] >= 5, "Should have at least OHLCV columns"

    X, y = engineer.create_sequences(feat)
    assert X.shape == (len(feat) - engineer.seq_len - 1, engineer.seq_len, feat.shape[1])
    assert y.shape[0] == X.shape[0]
    print("✅ feature_eng smoke-test passed")

def test_lstm_model():
    """Smoke-test LSTM forward pass."""
    batch   = 4
    seq_len = 60
    features = 17

    model = LSTMAttention(input_size=features, hidden_size=64,
                          num_layers=2, dropout=0.2)
    X = torch.randn(batch, seq_len, features)
    out = model(X)
    assert out.shape == (batch,)
    prob = model.predict_proba(X)
    assert (prob >= 0).all() and (prob <= 1).all()

    # Test LSTMModel wrapper
    lm = LSTMModel(input_size=features, device="cpu")
    p = lm.predict_proba(X)
    assert p.shape == (batch,)
    print("✅ lstm_model smoke-test passed")

def test_backtest():
    """Smoke-test backtest functions."""
    np.random.seed(0)
    probs  = np.random.uniform(0.3, 0.9, 200)
    actual = np.random.normal(0.001, 0.02, 200)

    result = backtest.backtest_strategy(probs, actual, threshold=0.55)
    assert "total_return" in result
    assert "sharpe"       in result
    assert "max_drawdown" in result

    bench = np.random.normal(0.0008, 0.015, 200)
    cmp = backtest.backtest_vs_benchmark(actual, bench)
    assert "alpha" in cmp
    print("✅ backtest smoke-test passed")

if __name__ == "__main__":
    test_feature_engineering()
    test_lstm_model()
    test_backtest()
    print("\n🎉 All tests passed!")