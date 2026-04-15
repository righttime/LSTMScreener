"""
추가 학습 (Fine-Tuning) 스크립트
- 기존 가중치 로드 (models/lstm_model_best.pt)
- 최근 60일 데이터로 fine-tuning (transfer learning)
- 학습률: 1e-4 (새 학습보다 낮게)
- 기존 모델 날짜 백업 → 새로 best 저장
- 에포크 15회 (빠르게)
"""
import os, sys, time, sqlite3, shutil, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DB     = os.path.join(PROJECT_DIR, "data", "market.db")
MODEL_DIR   = os.path.join(PROJECT_DIR, "models")
MODEL_BEST  = os.path.join(MODEL_DIR, "lstm_model_best.pt")

# ── hyperparameters ───────────────────────────────────────────────────────
SEQ_LEN    = 60
EPOCHS     = 15          # 빠르게 fine-tuning
BATCH_SIZE = 512
LR         = 1e-4        # 처음 학습(0.001)보다 낮게
DEVICE     = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
SEED       = 42
TARGET_GAIN = 0.01
RETRAIN_DAYS = 200       # 최근 N일 데이터만 사용 (SEQ_LEN+30=90 이상 保证)

print(f"Using device: {DEVICE} | Target Gain: {TARGET_GAIN*100}%")

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

# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [retrain.py] 추가 학습 (Fine-Tuning)")
print(f"  시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# 1. 기존 모델 백업
if os.path.exists(MODEL_BEST):
    date_str = datetime.now().strftime("%Y%m%d")
    backup_path = os.path.join(MODEL_DIR, f"lstm_model_backup_{date_str}.pt")
    shutil.copy2(MODEL_BEST, backup_path)
    print(f"\n✅ 기존 모델 백업 완료: {os.path.basename(backup_path)}")
else:
    print("\n[ERROR] 기존 모델을 찾을 수 없습니다: " + MODEL_BEST)
    sys.exit(1)

# 2. 기존 체크포인트 로드
checkpoint = torch.load(MODEL_BEST, weights_only=False)
print(f"✅ 모델 로드 완료 (학습된 epoch 수: 미상)")

# 3. DB에서 최근 데이터 로드
conn = sqlite3.connect(DATA_DB)
df_stocks = pd.read_sql("""
    SELECT code, name FROM stocks
    WHERE name NOT LIKE '%우'
      AND name NOT LIKE '%우B'
      AND name NOT LIKE '%우C'
""", conn)

valid_codes = df_stocks["code"].tolist()
placeholders = ','.join(['?'] * len(valid_codes))
df_candles = pd.read_sql(
    f"SELECT code, date, open, high, low, close, volume FROM candles WHERE code IN ({placeholders}) ORDER BY code, date",
    conn, params=valid_codes
)
conn.close()

print(f"  전체 유효 종목: {len(valid_codes)}개")

# 4. 최근 N일 데이터만 필터
latest_date = df_candles["date"].max()
cutoff_date = (datetime.strptime(latest_date, "%Y%m%d") - timedelta(days=RETRAIN_DAYS)).strftime("%Y%m%d")
print(f"  전체 데이터: {len(df_candles):,}건 | 필터: {cutoff_date} 이후 시퀀스 중심")

# 5. 시퀀스 생성
#    - Indicator 계산용: 전체 히스토리 사용
#    - Fine-tuning 대상: cutoff_date 이후 종료 시퀀스만
print(f"\n📊 시퀀스 생성 중 (전체 히스토리 → cutoff:{cutoff_date} 이후만 fine-tune)...")
all_X, all_y = [], []
for code in valid_codes:
    df_s = df_candles[df_candles["code"] == code].sort_values("date").reset_index(drop=True)
    if len(df_s) < SEQ_LEN + 30:
        continue
    df_feat = add_indicators(df_s).iloc[30:].reset_index(drop=True)
    df_feat["close"] = df_s["close"].iloc[30:].values
    df_feat["date"]  = df_s["date"].iloc[30:].values
    df_feat = df_feat.dropna().reset_index(drop=True)
    data_feats = np.nan_to_num(df_feat[ALL_COLS].values.astype(np.float32))
    data_close = df_feat["close"].values.astype(np.float32)
    data_dates = df_feat["date"].values
    # 시퀀스 중 끝 날짜가 cutoff_date 이후인 것만 fine-tuning에 사용
    for i in range(len(data_feats) - SEQ_LEN):
        seq_end_date = data_dates[i + SEQ_LEN - 1]
        if str(seq_end_date) < cutoff_date:
            continue  # 너무 오래된 시퀀스는 건너뜀
        ret_next = (data_close[i + SEQ_LEN] / data_close[i + SEQ_LEN - 1]) - 1
        label = 1.0 if ret_next >= TARGET_GAIN else 0.0
        all_X.append(data_feats[i: i + SEQ_LEN])
        all_y.append(label)

X = np.nan_to_num(np.array(all_X, dtype=np.float32))
y = np.array(all_y, dtype=np.float32)
print(f"  전체 샘플: {len(X):,} | 양성(1%↑): {y.mean()*100:.1f}%")

if len(X) < 100:
    print("[ERROR] 샘플이 너무 적습니다. DB 데이터를 확인하세요.")
    sys.exit(1)

# 6. 기존 mean/std 사용 (또는 새로 계산)
mean = checkpoint.get("mean")
std  = checkpoint.get("std")
if mean is None or std is None:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-5
else:
    # 기존 mean/std가 있으면 새 데이터 기준으로만 split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

X_train_n = np.nan_to_num((X_train - mean) / std)
X_val_n   = np.nan_to_num((X_val   - mean) / std)

train_ld = DataLoader(TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_ld   = DataLoader(TensorDataset(torch.from_numpy(X_val_n),   torch.from_numpy(y_val)),   batch_size=BATCH_SIZE, shuffle=False)

# 7. 모델 생성 + 기존 가중치 로드 (fine-tuning)
model = ResidualGRU(input_size=INPUT_SIZE).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.train()
print(f"\n🔧 Fine-tuning 시작 (lr={LR}, epochs={EPOCHS})")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_ld), epochs=EPOCHS)
pos_weight = torch.tensor([(1 - y.mean()) / y.mean()], dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_val_auc = 0
best_state   = None

# ── 학습 루프 ────────────────────────────────────────────────────────────
start_time = time.time()
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
    print(f"  Epoch {epoch+1:2d}/{EPOCHS} | loss: {t_loss/len(train_ld):.4f} | v_loss: {v_loss/len(val_ld):.4f} | val_auc: {auc:.4f}")

    if auc >= best_val_auc:
        best_val_auc = auc
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

elapsed = time.time() - start_time

# 8. 최적 모델 저장
torch.save({
    "model_state": best_state,
    "mean": mean,
    "std":  std,
}, MODEL_BEST)

# ── 최종 요약 ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ✅ 추가 학습 완료!")
print(f"     학습 시간: {elapsed:.1f}초")
print(f"     최종 val_auc: {best_val_auc:.4f}")
print(f"     모델 저장: {MODEL_BEST}")
print(f"{'='*60}")
