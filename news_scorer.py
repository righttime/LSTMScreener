"""
news_scorer.py — LSTM 스크리닝 결과에 뉴스 감성 점수를 부여하는 연동 모듈

AIStockAnalyzer의 뉴스 수집 + MiniMax 감성 분석을 활용하여
스크리닝된 종목에 뉴스 기반 감성 점수를 부여한다.
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# AIStockAnalyzer의 .env와 소스를 import할 수 있도록 경로 설정
AI_STOCK_DIR = Path("/root/project/AIStockAnalyzer")
sys.path.insert(0, str(AI_STOCK_DIR))

from dotenv import load_dotenv
load_dotenv(AI_STOCK_DIR / ".env")

from src.naver_client import search_naver_news_api
from src.news_fetcher import is_etf
from src.sentiment import batch_analyze

# ── 설정 ──────────────────────────────────────────────────────────────────────
NEWS_DAYS   = 3          # 최근 N일
NEWS_LIMIT  = 10         # 분석할 뉴스 수
TIMEOUT_SEC = 120        # 감성분석 타임아웃


def get_news_for_stock(name: str, code: str) -> list:
    """종목명으로 최근 뉴스 수집 (네이버 뉴스 API, 최근 3일 기준)"""
    if is_etf(name):
        # ETF: 검색용 이름 변환 시도
        from src.news_fetcher import get_etf_search_name
        query = get_etf_search_name(name) or name
    else:
        query = name

    results = search_naver_news_api(query, display=NEWS_LIMIT, sort="date")

    # 날짜 필터링 (최근 3일) — 올바른 pubDate 파싱
    # pubDate 형태: "Tue, 01 Apr 2025 10:30:00 +0900"
    cutoff = (datetime.now() - timedelta(days=NEWS_DAYS))
    filtered = []
    for item in results:
        pub = item.get("date", "")
        if not pub:
            continue
        try:
            # "DD Mon YYYY HH:MM:SS +0900" → datetime 파싱
            from email.utils import parsedate_to_datetime
            pub_dt = parsedate_to_datetime(pub)
            if pub_dt >= cutoff:
                filtered.append(item)
        except Exception:
            # 파싱 실패 시的新闻そのまま 포함（sort="date"로 최신 우선 정렬되어 있으므로）
            filtered.append(item)

    # 중복 제거 (URL 기준)
    seen, unique = set(), []
    for item in filtered:
        url = item.get("url", "")
        if url not in seen:
            seen.add(url)
            unique.append(item)

    return unique[:NEWS_LIMIT]


def compute_news_score(name: str, code: str) -> float:
    """
    종목의 뉴스 감성 점수를 계산하여 -1.0 ~ 1.0 사이 값 반환.
    오류 시 0.0 (중립) 반환.
    """
    try:
        news_list = get_news_for_stock(name, code)
        if not news_list:
            print(f"  [news_scorer] {name}({code}): 뉴스 없음 → 0.0 (중립)")
            return 0.0

        # 감성분석 (병렬, 타임아웃 내부 처리)
        analyzed = batch_analyze(news_list[:NEWS_LIMIT], name, code)

        positive = sum(1 for n in analyzed if n.get("sentiment") == "positive")
        negative = sum(1 for n in analyzed if n.get("sentiment") == "negative")
        neutral  = sum(1 for n in analyzed if n.get("sentiment")  == "neutral")
        total    = positive + negative + neutral or 1

        # 점수 계산: (긍정 - 부정) / 전체 * 1.0  + 중립 고려
        raw_score = sum(n.get("score", 0.0) for n in analyzed) / total

        # -1 ~ 1 범위 보장
        score = max(-1.0, min(1.0, raw_score))

        pos_pct = positive / total * 100
        neg_pct = negative / total * 100
        print(f"  [news_scorer] {name}({code}): 긍정 {pos_pct:.0f}% 부정 {neg_pct:.0f}% → {score:+.3f}")

        return score

    except Exception as e:
        print(f"  [news_scorer] {name}({code}): 오류 '{e}' → 0.0 (중립)", flush=True)
        return 0.0


def normalize_sentiment(score: float) -> float:
    """
    감성 점수 -1.0~1.0 → 0.0~1.0 정규화 (강한 스케일링으로 차별화)
    점수에 3x gain 적용 후 클램핑 → 중립 근처 더 민감, 극단 더 극단
    예: +0.1→0.65, +0.3→0.95, -0.1→0.35, -0.3→0.05
    """
    boosted = score * 3.0
    boosted = max(-1.0, min(1.0, boosted))
    return (boosted + 1.0) / 2.0
