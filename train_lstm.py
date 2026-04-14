"""
Targeting 1% Growth: Including Stocks & ETFs (Excluding Preferred Stocks)
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
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DB     = os.path.join(PROJECT_DIR, "data", "market.db")
MODEL_DIR   = os.path.join(PROJECT_DIR, "models")

# ── hyperparameters ───────────────────────────────────────────────────────
SEQ_LEN    = 60
EPOCHS     = 200
BATCH_SIZE = 512
LR         = 0.001
DEVICE     = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
SEED       = 42
TARGET_GAIN = 0.01 # Still target at least 1% gain

print(f"Using device: {DEVICE} | Target Gain: {TARGET_GAIN*100}%")
os.makedirs(MODEL_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

ALL_COLS = [
    "ret", "vol_ret", "rsi", "macd_rel", "ma5_rel", "ma20_rel", "ma60_rel",
    "bb_width", "volatility", "hl_ratio", "oc_ratio", "upper_shadow", "lower_shadow", "vol_ma_rel"
]
INPUT_SIZE = len(ALL_COLS)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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

class ResidualGRU(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=128, num_layers=2, dropout=0.5):
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
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

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

print("STEP 1: Load Stock & ETF data (Excluding Preferred Stocks)")
conn = sqlite3.connect(DATA_DB)
df_stocks = pd.read_sql("""
    SELECT code, name FROM stocks 
    WHERE name NOT LIKE '%우'
      AND name NOT LIKE '%우B'
      AND name NOT LIKE '%우C'
""", conn)
valid_codes = df_stocks["code"].tolist()
placeholders = ','.join(['?'] * len(valid_codes))
df_candles = pd.read_sql(f"SELECT code, date, open, high, low, close, volume FROM candles WHERE code IN ({placeholders}) ORDER BY code, date", conn, params=valid_codes)
conn.close()

print(f"  Total entities after filtering: {len(valid_codes)}")

print("STEP 2: Build sequences (Label: Gain >= 1%)")
all_X, all_y = [], []
for code in valid_codes:
    df_s = df_candles[df_candles["code"] == code].sort_values("date").reset_index(drop=True)
    if len(df_s) < SEQ_LEN + 30: continue
    df_feat = add_indicators(df_s).iloc[30:].reset_index(drop=True)
    df_feat["close"] = df_s["close"].iloc[30:].values
    df_feat = df_feat.dropna().reset_index(drop=True)
    data_feats = np.nan_to_num(df_feat[ALL_COLS].values.astype(np.float32))
    data_close = df_feat["close"].values.astype(np.float32)
    for i in range(len(data_feats) - SEQ_LEN):
        ret_next = (data_close[i + SEQ_LEN] / data_close[i + SEQ_LEN - 1]) - 1
        label = 1.0 if ret_next >= TARGET_GAIN else 0.0
        all_X.append(data_feats[i : i + SEQ_LEN])
        all_y.append(label)

X = np.nan_to_num(np.array(all_X, dtype=np.float32))
y = np.array(all_y, dtype=np.float32)
print(f"  Total samples: {len(X):,}, Positive (>=1% gain): {y.mean()*100:.1f}%")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
mean = X_train.mean(axis=(0, 1), keepdims=True)
std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-5
X_train_n = np.nan_to_num((X_train - mean) / std)
X_val_n   = np.nan_to_num((X_val   - mean) / std)

train_ld = DataLoader(TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_ld   = DataLoader(TensorDataset(torch.from_numpy(X_val_n),   torch.from_numpy(y_val)),   batch_size=BATCH_SIZE, shuffle=False)

print(f"STEP 3: Training (Stocks + ETFs | Target: 1% Gain)")
model = ResidualGRU().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_ld), epochs=EPOCHS)
pos_weight = torch.tensor([(1 - y.mean()) / y.mean()], dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_val_auc = 0
best_state = None

for epoch in range(EPOCHS):
    model.train()
    t_loss = 0
    for Xb, yb in train_ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        yb_smooth = yb * 0.9 + 0.05
        loss = criterion(model(Xb), yb_smooth)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        t_loss += loss.item()
    
    model.eval()
    v_loss, v_preds, v_labs = 0, [], []
    with torch.no_grad():
        for Xb, yb in val_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            out = model(Xb)
            v_loss += criterion(out, yb).item()
            v_preds.extend(torch.sigmoid(out).cpu().numpy())
            v_labs.extend(yb.cpu().numpy())
    
    v_preds_arr = np.array(v_preds)
    auc = roc_auc_score(v_labs, v_preds_arr)
    print(f"  Epoch {epoch+1:3d} | loss: {t_loss/len(train_ld):.4f} | v_loss: {v_loss/len(val_ld):.4f} | auc: {auc:.4f}")

    if auc > best_val_auc:
        best_val_auc = auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"model_state": best_state, "mean": mean, "std": std}, os.path.join(MODEL_DIR, "lstm_model_best.pt"))

print(f"\nTraining Complete. Best val_auc: {best_val_auc:.4f}")
