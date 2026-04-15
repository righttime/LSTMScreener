#!/usr/bin/env python3
"""
screen_with_news.py — LSTM 확률 × 뉴스 감성 점수 통합 스크리닝

LSTMScreener의 LSTM 모델로 상위 N개 종목을 스크리닝한 뒤,
AIStockAnalyzer의 뉴스 감성 분석을 결합하여 최종 순위를 산출한다.

최종 점수 = LSTM 확률(0~1) × 0.6 + 뉴스 감성 정규화(0~1) × 0.4

출력 예시:
  순위 | 코드    | 종목      | LSTM  | 뉴스  | 최종
  ─────────────────────────────────────────────
    1  | 005930 | 삼성전자  | 0.842 | 0.73  | 0.797
    2  | 000660 | SK하이닉스 | 0.815 | 0.82  | 0.817
"""
import os
import sys
import json
import sqlite3
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── LSTM 모델 아키텍처 (screen_stocks.py와 동일) ──────────────────────────────
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
            if i > 0:
                out = out + x
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
    df["ret"]       = close.pct_change().fillna(0)
    df["vol_ret"]   = vol.pct_change().fillna(0)
    for w in [5, 20, 60]:
        ma = close.rolling(w).mean()
        df[f"ma{w}_rel"] = (close / ma - 1).fillna(0)
    df["hl_ratio"]     = ((high - low) / close).fillna(0)
    df["oc_ratio"]     = ((close - open_p) / close).fillna(0)
    df["upper_shadow"] = ((high - df[["open", "close"]].max(axis=1)) / close).fillna(0)
    df["lower_shadow"] = ((df[["open", "close"]].min(axis=1) - low) / close).fillna(0)
    vol_ma20 = vol.rolling(20).mean()
    df["vol_ma_rel"]   = (vol / vol_ma20 - 1).fillna(0)
    delta = close.diff()
    gain, loss = (delta.where(delta > 0, 0).rolling(14).mean(),
                  (-delta.where(delta < 0, 0)).rolling(14).mean())
    rs = gain / loss.replace(0, np.nan)
    df["rsi"]        = (100 - (100 / (1 + rs.fillna(0)))).fillna(50) / 100.0
    ema_fast, ema_slow = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df["macd_rel"]   = ((ema_fast - ema_slow) / close).fillna(0)
    bb_mid, bb_std  = close.rolling(20).mean(), close.rolling(20).std()
    df["bb_width"]   = ((bb_std * 4) / bb_mid).fillna(0)
    df["volatility"] = df["ret"].rolling(20).std().fillna(0)
    return df


# ── 설정 ──────────────────────────────────────────────────────────────────────
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DB      = os.path.join(PROJECT_DIR, "data", "market.db")
MODEL_PATH   = os.path.join(PROJECT_DIR, "models", "lstm_model_best.pt")
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"
SEQ_LEN      = 60
ALL_COLS     = ["ret", "vol_ret", "rsi", "macd_rel",
                "ma5_rel", "ma20_rel", "ma60_rel",
                "bb_width", "volatility",
                "hl_ratio", "oc_ratio",
                "upper_shadow", "lower_shadow", "vol_ma_rel"]

TOP_N          = 30     # LSTM 스크리닝 후보 수
NEWS_TOP_N     = 30     # 뉴스 분석 대상 수 (0이면 전체)
TEST_MODE      = True   # True: TOP 5만 뉴스 분석 (빠른 테스트)
LSTM_WEIGHT    = 0.7
NEWS_WEIGHT    = 0.3

# ── LSTM 모델 로드 ─────────────────────────────────────────────────────────────
print("=" * 60, flush=True)
print(f">>> LSTM 모델 로드 중... (device: {DEVICE})", flush=True)
checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
model = ResidualGRU(input_size=len(ALL_COLS)).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()
mean, std  = checkpoint["mean"], checkpoint["std"]
device_mean = torch.from_numpy(mean).to(DEVICE)
device_std  = torch.from_numpy(std).to(DEVICE)
print(">>> LSTM 모델 로드 완료", flush=True)

# ── DB에서 종목 데이터 로드 ───────────────────────────────────────────────────
conn = sqlite3.connect(DATA_DB)
df_candles = pd.read_sql(
    "SELECT code, date, open, high, low, close, volume FROM candles ORDER BY code, date", conn)
df_names   = pd.read_sql("SELECT code, name FROM stocks", conn)
conn.close()
name_map = dict(zip(df_names["code"], df_names["name"]))

# ── LSTM 스크리닝 ─────────────────────────────────────────────────────────────
print("\n>>> LSTM 스크리닝 실행...", flush=True)
results = []
stock_codes = df_candles["code"].unique()
for idx, code in enumerate(stock_codes):
    df_s = df_candles[df_candles["code"] == code].sort_values("date").reset_index(drop=True)
    if len(df_s) < SEQ_LEN + 30:
        continue
    df_feat = add_indicators(df_s).dropna()
    if len(df_feat) < SEQ_LEN:
        continue
    last_seq = df_feat[ALL_COLS].values[-SEQ_LEN:].astype(np.float32)
    with torch.no_grad():
        seq_t = torch.from_numpy(last_seq).to(DEVICE).unsqueeze(0)
        seq_n = (seq_t - device_mean) / device_std
        prob  = torch.sigmoid(model(seq_n)).item()
        results.append((code, prob))
    if (idx + 1) % 200 == 0:
        print(f"  [{idx+1}/{len(stock_codes)}] 처리 완료", flush=True)

results.sort(key=lambda x: x[1], reverse=True)
print(f">>> LSTM 스크리닝 완료: {len(results)}개 종목을 평가함", flush=True)

# TOP N 후보
candidates = results[:TOP_N] if TOP_N > 0 else results
if TEST_MODE:
    candidates = candidates[:5]
    print(f">>> [테스트 모드] TOP {len(candidates)}개만 뉴스 분석 진행", flush=True)
else:
    print(f">>> TOP {len(candidates)}개 종목에 뉴스 감성 분석 적용", flush=True)

# ── 뉴스 감성 분석 연동 ────────────────────────────────────────────────────────
from news_scorer import compute_news_score, normalize_sentiment

print("\n>>> 뉴스 감성 분석 시작...", flush=True)
scored = []
total_time = 0

for rank, (code, lstm_prob) in enumerate(candidates, 1):
    name = name_map.get(code, code)
    start = time.time()
    news_score_raw = compute_news_score(name, code)
    elapsed = time.time() - start
    total_time += elapsed

    news_score_n = normalize_sentiment(news_score_raw)
    final_score  = lstm_prob * LSTM_WEIGHT + news_score_n * NEWS_WEIGHT

    scored.append({
        "rank":        rank,
        "code":        code,
        "name":        name,
        "lstm_prob":   lstm_prob,
        "news_score":  news_score_n,
        "news_raw":    news_score_raw,
        "final_score": final_score,
        "time_sec":    round(elapsed, 1),
    })
    print(f"  {rank:>2}. {code} {name:<15} LSTM={lstm_prob:.4f} 뉴스={news_score_n:.3f}({news_score_raw:+.2f}) 최종={final_score:.4f} ({elapsed:.1f}s)")

# ── 최종 순위 정렬 ────────────────────────────────────────────────────────────
scored.sort(key=lambda x: x["final_score"], reverse=True)
for i, s in enumerate(scored, 1):
    s["rank"] = i

# ── 결과 출력 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 75, flush=True)
print(f"  {'순위':>3} | {'코드':<10} | {'종목':<18} | {'LSTM':>6} | {'뉴스':>5} | {'최종':>6} | {'추천':>5}", flush=True)
print(f"  {'-'*72}", flush=True)

for s in scored:
    # LSTM + 뉴스 조합으로 BUY/HOLD/AVOID 판단
    if s["final_score"] >= 0.70:
        rec = "✅ BUY"
    elif s["final_score"] >= 0.55:
        rec = "🟡 HOLD"
    else:
        rec = "🔴 AVOID"

    print(f"  {s['rank']:>3} | {s['code']:<10} | {s['name']:<18} | "
          f"{s['lstm_prob']:>6.4f} | {s['news_score']:>5.3f} | "
          f"{s['final_score']:>6.4f} | {rec}", flush=True)

print(f"  {'-'*72}", flush=True)
print(f"  총 {len(scored)}개 종목 | 총 소요시간 {total_time:.1f}초", flush=True)
print(f"  최종 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print("=" * 75, flush=True)

# ── CSV 저장 ─────────────────────────────────────────────────────────────────
out_csv = os.path.join(PROJECT_DIR, "data", "screen_with_news.csv")
df_out = pd.DataFrame(scored)
df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\n📄 결과 저장: {out_csv}")

# ── prev_close 조회 ─────────────────────────────────────────────────────────
# DB에서 각 종목의 직전 영업일 종가를 가져온다
code_last_candle = {}
for code in df_candles["code"].unique():
    rows = df_candles[df_candles["code"] == code].sort_values("date")
    if len(rows) >= 2:
        code_last_candle[code] = int(rows.iloc[-2]["close"])
    elif len(rows) == 1:
        code_last_candle[code] = int(rows.iloc[-1]["close"])

# ── 와치리스트 생성 (LSTM 10 + TA 10 + combined 20) ────────────────────────
LSTM_TOP = 10
TA_TOP = 10
now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+09:00")

# LSTM top 10: pure LSTM probability 기준
scored_by_lstm = sorted(scored, key=lambda x: x["lstm_prob"], reverse=True)
lstm_top10 = scored_by_lstm[:LSTM_TOP]
lstm_watchlist = {
    "updated": now_str,
    "stocks": [
        {
            "code": s["code"],
            "name": s["name"],
            "score": round(s["lstm_prob"], 4),
            "source": "lstm",
            "prev_close": code_last_candle.get(s["code"], 0),
        }
        for s in lstm_top10
    ]
}
lstm_path = os.path.join(PROJECT_DIR, "data", "watchlist_lstm.json")
with open(lstm_path, "w", encoding="utf-8") as f:
    json.dump(lstm_watchlist, f, ensure_ascii=False, indent=2)
print(f"\n📋 LSTM 와치리스트 저장: {lstm_path} ({len(lstm_top10)}종목)")

# TA top 10: final_score (LSTM+news) 기준
scored_by_final = sorted(scored, key=lambda x: x["final_score"], reverse=True)
ta_top10 = scored_by_final[:TA_TOP]
ta_watchlist = {
    "updated": now_str,
    "stocks": [
        {
            "code": s["code"],
            "name": s["name"],
            "score": round(s["final_score"], 4),
            "source": "ta",
            "prev_close": code_last_candle.get(s["code"], 0),
        }
        for s in ta_top10
    ]
}
ta_path = os.path.join(PROJECT_DIR, "data", "watchlist_ta.json")
with open(ta_path, "w", encoding="utf-8") as f:
    json.dump(ta_watchlist, f, ensure_ascii=False, indent=2)
print(f"📋 TA 와치리스트 저장: {ta_path} ({len(ta_top10)}종목)")

# Combined 20: LSTM 10 + TA 10 (중복 허용, 서로 다른 source)
combined_stocks = []
for s in lstm_top10:
    combined_stocks.append({
        "code": s["code"],
        "name": s["name"],
        "score": round(s["lstm_prob"], 4),
        "source": "lstm",
        "prev_close": code_last_candle.get(s["code"], 0),
    })
for s in ta_top10:
    combined_stocks.append({
        "code": s["code"],
        "name": s["name"],
        "score": round(s["final_score"], 4),
        "source": "ta",
        "prev_close": code_last_candle.get(s["code"], 0),
    })

combined_watchlist = {
    "updated": now_str,
    "stocks": combined_stocks
}
watchlist_path = os.path.join(PROJECT_DIR, "data", "watchlist.json")
with open(watchlist_path, "w", encoding="utf-8") as f:
    json.dump(combined_watchlist, f, ensure_ascii=False, indent=2)
print(f"📋 Combined 와치리스트 저장: {watchlist_path} ({len(combined_stocks)}종목)")
