# LSTMScreener
# LSTM 기반 주식 스크리닝 + MiniMax LLM 펀멘탈 분석 하이브리드

## 개요
- **LSTM 모델**: PyTorch 기반, 과거 60일 시퀀스 → 다음 날 등락 확률 예측
- **LLM 분석**: MiniMax API로 펀멘탈 분석 (재무제표, 뉴스, 업종)
- **하이브리드 스코어**: LSTM 60% + LLM 40%

## 구조
```
LSTMScreener/
├── src/
│   ├── config.py        # 설정 (API 키, 경로, 파라미터)
│   ├── data_loader.py    # 키움/야후 데이터 로딩 + 캐시
│   ├── feature_eng.py   # 기술적 지표 + 시퀀스 생성
│   ├── lstm_model.py    # LSTM + Attention 모델
│   ├── llm_analyzer.py  # MiniMax 펀멘탈 분석
│   ├── screener.py      # 전종목 스크리닝 + 최종 추천
│   └── backtest.py      # 백테스트
├── models/              # 학습된 모델 (.pt)
├── data/                # 캐시된 데이터 (.parquet)
├── tests/
│   └── test_model.py
└── notebooks/           # 분석용 Jupyter 노트북
```

## 빠른 시작
```bash
pip install -r requirements.txt

# 1. 데이터 다운로드
python -m src.data_loader --download

# 2. 모델 학습
python -m src.lstm_model --train

# 3. 전종목 스크리닝
python -m src.screener --screen --top 20

# 4. 백테스트
python -m src.backtest
```

## 하이브리드 스코어 공식
```
final_score = lstm_prob * 0.6 + llm_score * 0.4
```
- LSTM 확률 0~1 (등락 예측 신뢰도)
- LLM 점수 -1~1 (매수=+1, 보류=0, 매도=-1 → 정규화)

## 참고
- GPU 없어도 CPU로 동작
- 키움 API 키 필요 (실전 or 모의)
- MiniMax API 키 필요
