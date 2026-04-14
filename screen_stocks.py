"""
Screening stocks using the best trained model
"""
import os, sqlite3, torch
import numpy as np
import pandas as pd
import torch.nn as nn

# Same model architecture as train_lstm.py
class ResidualGRU(nn.Module):
    def __init__(self, input_size=14, hidden_size=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.ln_in = nn.LayerNorm(input_size)
        self.grus = nn.ModuleList([
            nn.GRU(input_size if i == 0 else hidden_size, hidden_size, batch_first=True)
            for i in range(num_layers)
        ])
        self.ln_mid = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_size, 1, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.ln_in(x)
        h_state = None
        for i, gru in enumerate(self.grus):
            out, _ = gru(x, h_state)
            if i > 0: out = out + x
            out = self.ln_mid[i](out)
            x = self.dropout(out)
        scores = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(1)
        context = torch.bmm(weights, x).squeeze(1)
        return self.fc(context).squeeze(-1)

def add_indicators(df):
    df = df.copy()
    close = df["close"].replace(0, np.nan)
    high, low, open_p, vol = df["high"], df["low"], df["open"], df["volume"].replace(0, np.nan)
    df["ret"] = close.pct_change().fillna(0)
    df["vol_ret"] = vol.pct_change().fillna(0)
    for w in [5, 20, 60]:
        ma = close.rolling(w).mean()
        df[f"ma{w}_rel"] = (close / ma - 1).fillna(0)
    df["hl_ratio"] = ((high - low) / close).fillna(0)
    df["oc_ratio"] = ((close - open_p) / close).fillna(0)
    df["upper_shadow"] = ((high - df[["open", "close"]].max(axis=1)) / close).fillna(0)
    df["lower_shadow"] = ((df[["open", "close"]].min(axis=1) - low) / close).fillna(0)
    vol_ma20 = vol.rolling(20).mean()
    df["vol_ma_rel"] = (vol / vol_ma20 - 1).fillna(0)
    delta = close.diff()
    gain, loss = delta.where(delta > 0, 0).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs.fillna(0)))).fillna(50) / 100.0
    ema_fast, ema_slow = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df["macd_rel"] = ((ema_fast - ema_slow) / close).fillna(0)
    bb_mid, bb_std = close.rolling(20).mean(), close.rolling(20).std()
    df["bb_width"] = ((bb_std * 4) / bb_mid).fillna(0)
    df["volatility"] = df["ret"].rolling(20).std().fillna(0)
    return df

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DB     = os.path.join(PROJECT_DIR, "data", "market.db")
MODEL_PATH  = os.path.join(PROJECT_DIR, "models", "lstm_model_best.pt")
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
SEQ_LEN     = 60
ALL_COLS    = ["ret", "vol_ret", "rsi", "macd_rel", "ma5_rel", "ma20_rel", "ma60_rel", "bb_width", "volatility", "hl_ratio", "oc_ratio", "upper_shadow", "lower_shadow", "vol_ma_rel"]

print(">>> Loading best model and data...")
checkpoint = torch.load(MODEL_PATH, weights_only=False)
model = ResidualGRU(input_size=len(ALL_COLS)).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

mean, std = checkpoint["mean"], checkpoint["std"]
device_mean, device_std = torch.from_numpy(mean).to(DEVICE), torch.from_numpy(std).to(DEVICE)

conn = sqlite3.connect(DATA_DB)
df_candles = pd.read_sql("SELECT code, date, open, high, low, close, volume FROM candles ORDER BY code, date", conn)
df_names = pd.read_sql("SELECT code, name FROM stocks", conn)
conn.close()
name_map = dict(zip(df_names["code"], df_names["name"]))

print(">>> Screening stocks...")
results = []
stock_codes = df_candles["code"].unique()
for code in stock_codes:
    df_s = df_candles[df_candles["code"] == code].sort_values("date").reset_index(drop=True)
    if len(df_s) < SEQ_LEN + 30: continue
    df_feat = add_indicators(df_s).dropna()
    if len(df_feat) < SEQ_LEN: continue
    
    last_seq = df_feat[ALL_COLS].values[-SEQ_LEN:].astype(np.float32)
    with torch.no_grad():
        seq_t = torch.from_numpy(last_seq).to(DEVICE).unsqueeze(0)
        seq_n = (seq_t - device_mean) / device_std
        prob = torch.sigmoid(model(seq_n)).item()
        results.append((code, prob))

results.sort(key=lambda x: x[1], reverse=True)
print(f"\n  {'Rank':<5} {'Code':<10} {'Name':<20} {'Up Prob':>8}")
print(f"  {'-'*47}")
for rank, (code, prob) in enumerate(results[:10], 1):
    print(f"  {rank:<5} {code:<10} {name_map.get(code,'Unknown'):<20} {prob:.4f}")
