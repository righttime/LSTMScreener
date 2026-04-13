"""
Train LSTM model from TraderAlfred market.db
"""
import os, sys, time, sqlite3, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR = "/root/project/LSTMScreener"
DATA_DB     = "/root/project/TraderAlfred/data/market.db"
MODEL_DIR   = os.path.join(PROJECT_DIR, "models")

# ── hyperparameters ───────────────────────────────────────────────────────
SEQ_LEN    = 60
EPOCHS     = 30
BATCH_SIZE = 64
LR         = 0.001
DEVICE     = "cpu"
SEED       = 42

# ── feature columns (21 total) ─────────────────────────────────────────────
BASE_COLS  = ["open","high","low","close","volume"]
FEAT_COLS  = ["ma5","ma20","ma60","ma120",
              "rsi","macd","macd_signal","macd_hist",
              "bb_upper","bb_lower","bb_width","vol_ratio",
              "momentum_5","momentum_20","return_1d","volatility_20"]
ALL_COLS   = BASE_COLS + FEAT_COLS   # 21 columns
INPUT_SIZE = len(ALL_COLS)

os.makedirs(MODEL_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── technical indicators ────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    for w in [5, 20, 60, 120]:
        df[f"ma{w}"] = close.rolling(w).mean()
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid
    df["vol_ma20"]  = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)
    df["momentum_5"]  = close / close.shift(5)  - 1
    df["momentum_20"] = close / close.shift(20) - 1
    df["return_1d"]   = close.pct_change()
    df["volatility_20"] = df["return_1d"].rolling(20).std()
    return df

# ── LSTM model ─────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attn = nn.Linear(hidden_size, 1, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)                     # (B, SEQ_LEN, hidden)
        scores = self.attn(out).squeeze(-1)        # (B, SEQ_LEN)
        w = torch.softmax(scores, dim=1).unsqueeze(1)  # (B, 1, SEQ_LEN)
        ctx = torch.bmm(w, out).squeeze(1)        # (B, hidden)
        return self.fc(ctx).squeeze(-1)           # (B,)

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Load data from market.db")
print("=" * 60)
t_start = time.time()
conn = sqlite3.connect(DATA_DB)
df_candles = pd.read_sql(
    "SELECT code, date, open, high, low, close, volume FROM candles ORDER BY code, date",
    conn
)
conn.close()
print(f"  Loaded {len(df_candles):,} rows in {time.time()-t_start:.1f}s")
print(f"  Stocks: {df_candles['code'].nunique()}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Build sequences (SEQ_LEN=60, horizon=1)")
print("=" * 60)
t2 = time.time()
all_X, all_y = [], []
stock_codes = df_candles["code"].unique()

for code in stock_codes:
    df_s = df_candles[df_candles["code"] == code].copy()
    df_s["date"] = pd.to_datetime(df_s["date"])
    df_s = df_s.sort_values("date").reset_index(drop=True)
    df_s = add_indicators(df_s)

    # fill any missing feature columns with 0
    for col in ALL_COLS:
        if col not in df_s.columns:
            df_s[col] = 0.0

    data = df_s[ALL_COLS].values.astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    for i in range(len(data) - SEQ_LEN - 1):
        price_now  = data[i + SEQ_LEN - 1, 3]   # close at end of window
        price_next = data[i + SEQ_LEN, 3]         # next-day close
        label = 1.0 if price_next >= price_now else 0.0
        all_X.append(data[i : i + SEQ_LEN])
        all_y.append(label)

X = np.array(all_X, dtype=np.float32)
y = np.array(all_y, dtype=np.float32)
print(f"  Built {len(X):,} samples in {time.time()-t2:.1f}s")
print(f"  Label=1 (up):   {int(y.sum()):,} ({y.mean()*100:.1f}%)")
print(f"  Label=0 (down): {int(len(y)-y.sum()):,} ({(1-y.mean())*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Train/Val split (80/20)")
print("=" * 60)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True
)
print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")

# Normalize: fit on train only
mean = X_train.mean(axis=(0, 1), keepdims=True)
std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
X_train_n = (X_train - mean) / std
X_val_n   = (X_val   - mean) / std

train_ds = TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val_n),   torch.from_numpy(y_val))
train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 4: Training (epochs={EPOCHS}, batch={BATCH_SIZE})")
print("=" * 60)
model = LSTMModel(input_size=INPUT_SIZE).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR)
crit  = nn.BCEWithLogitsLoss()

t_train = time.time()
best_val_loss = float("inf")
best_state    = None

for epoch in range(EPOCHS):
    model.train()
    t_loss = 0.0
    for Xb, yb in train_ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(Xb), yb)
        loss.backward()
        opt.step()
        t_loss += loss.item()
    t_loss /= len(train_ld)

    model.eval()
    v_loss = 0.0
    v_preds, v_labs = [], []
    with torch.no_grad():
        for Xb, yb in val_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            out = model(Xb)
            v_loss += crit(out, yb).item()
            v_preds.extend(torch.sigmoid(out).cpu().numpy())
            v_labs.extend(yb.cpu().numpy())
    v_loss /= len(val_ld)

    acc = accuracy_score(v_labs, (np.array(v_preds) >= 0.5).astype(int))
    auc = roc_auc_score(v_labs, v_preds)
    print(f"  Epoch {epoch+1:2d}/{EPOCHS} | train_loss: {t_loss:.4f} | val_loss: {v_loss:.4f} | acc: {acc:.4f} | auc: {auc:.4f}")

    if v_loss < best_val_loss:
        best_val_loss = v_loss
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

elapsed_train = time.time() - t_train
print(f"\n  Training done in {elapsed_train:.0f}s ({elapsed_train/60:.1f} min)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Best model metrics")
print("=" * 60)
model.load_state_dict(best_state)
model.eval()

final_preds, final_labels = [], []
with torch.no_grad():
    for Xb, yb in val_ld:
        out = torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy()
        final_preds.extend(out)
        final_labels.extend(yb.cpu().numpy())

final_preds  = np.array(final_preds)
final_labels = np.array(final_labels)
final_acc    = accuracy_score(final_labels, (final_preds >= 0.5).astype(int))
final_auc    = roc_auc_score(final_labels, final_preds)
print(f"  Best val_loss:  {best_val_loss:.4f}")
print(f"  Final accuracy: {final_acc:.4f}")
print(f"  Final AUC:      {final_auc:.4f}")

# Save model
model_path = os.path.join(MODEL_DIR, "lstm_model.pt")
torch.save({
    "model_state":  best_state,
    "mean":         mean,
    "std":          std,
    "config": {
        "input_size":   INPUT_SIZE,
        "hidden_size":  128,
        "num_layers":   2,
        "dropout":      0.3,
        "seq_len":      SEQ_LEN,
    },
    "metrics": {
        "val_loss":     float(best_val_loss),
        "accuracy":     float(final_acc),
        "auc":          float(final_auc),
        "epochs_run":   EPOCHS,
    },
}, model_path)
print(f"  Model saved → {model_path}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Screening — Top 10 by LSTM up-probability")
print("=" * 60)
t_scr = time.time()

stock_last = {}
for code in stock_codes:
    df_s = df_candles[df_candles["code"] == code].copy()
    df_s["date"] = pd.to_datetime(df_s["date"])
    df_s = df_s.sort_values("date").reset_index(drop=True)
    if len(df_s) < SEQ_LEN + 10:
        continue
    df_s = add_indicators(df_s)
    for col in ALL_COLS:
        if col not in df_s.columns:
            df_s[col] = 0.0
    data = df_s[ALL_COLS].values.astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    stock_last[code] = data[-SEQ_LEN:]

print(f"  Screening {len(stock_last)} stocks ...")
model.eval()
results = []
with torch.no_grad():
    for code, seq in stock_last.items():
        seq_n = (seq - mean) / std
        prob  = torch.sigmoid(model(torch.from_numpy(seq_n[np.newaxis,:]).to(DEVICE)).squeeze(-1)).item()
        results.append((code, prob))

results.sort(key=lambda x: x[1], reverse=True)
top10 = results[:10]

conn = sqlite3.connect(DATA_DB)
df_names = pd.read_sql("SELECT code, name FROM stocks", conn)
conn.close()
name_map = dict(zip(df_names["code"], df_names["name"]))

print(f"\n  {'Rank':<5} {'Code':<10} {'Name':<22} {'Prob':>6}")
print(f"  {'-'*47}")
for rank, (code, prob) in enumerate(top10, 1):
    name = name_map.get(code, "Unknown")
    print(f"  {rank:<5} {code:<10} {name:<22} {prob:.4f}")

print(f"\n  Screening done in {time.time()-t_scr:.1f}s")
total = time.time() - t_start
print(f"\n✅ TOTAL TIME: {total:.0f}s ({total/60:.1f} min)")
print(f"   Model saved: {model_path}")