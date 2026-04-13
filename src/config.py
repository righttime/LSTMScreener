from __future__ import annotations
"""Configuration for LSTMScreener."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"

# API Keys (set via environment or .env file)
KIWOOM_API_KEY = os.getenv("KIWOOM_API_KEY", "")
KIWOOM_SECRET_KEY = os.getenv("KIWOOM_SECRET_KEY", "")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")

# Kiwoom API endpoints
KIWOOM_BASE_URL = "https://openapi.kiwoom.com"
KIWOOM_STOCK_LIST_URL = f"{KIWOOM_BASE_URL}/api/dostk/ths/GetTHSnsmartstddaybassiseeksshcode"

# Model hyperparameters
SEQ_LEN = 60          # 입력 시퀀스 길이 (일)
HIDDEN_SIZE = 128     # LSTM 은닉 크기
NUM_LAYERS = 2        # LSTM 레이어 수
DROPOUT = 0.3         # 드롭아웃 비율
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 50
EARLY_STOP_PATIENCE = 10

# Feature Engineering
MA_WINDOWS = [5, 20, 60, 120]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2

# Scoring weights
LSTM_WEIGHT = 0.6
LLM_WEIGHT = 0.4
