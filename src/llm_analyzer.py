from __future__ import annotations
"""MiniMax LLM fundamental analysis."""
import os
import httpx
from typing import Literal
from src import config

LLM_ENDPOINT = "https://api.minimax.chat/v1/text/chatcompletion_pro"

# LLM score mapping
LLM_SCORES = {"매수": 1.0, "보류": 0.5, "매도": 0.0}

def analyze_fundamental(ticker: str, company_name: str,
                         financials: dict = None,
                         news: list = None,
                         sector_news: str = "") -> dict:
    """Run fundamental analysis via MiniMax and return recommendation.

    Params
    ------
    ticker        : Stock code
    company_name  : Company name
    financials    : Dict of financial metrics (revenue, PER, PBR, ROE, etc.)
    news          : List of recent news headlines/summaries
    sector_news   : String describing sector-wide news

    Returns
    -------
    dict with keys: recommendation (str), score (float 0~1), reason (str)
    """
    if not config.MINIMAX_API_KEY:
        return {"recommendation": "보류", "score": 0.5,
                "reason": "MINIMAX_API_KEY not set"}

    # Build prompt
    prompt_lines = [
        f"종목: {company_name} ({ticker})",
        f"재무지표: {financials or 'N/A'}",
        "",
        f"종목 관련 뉴스:\n{chr(10).join(news) if news else '相关新闻 없음'}",
        "",
        f"업종 동향:\n{sector_news}",
        "",
        "위 정보를 바탕으로 다음을 판단해줘:\n"
        "1) 매수/보류/매도 중 하나\n"
        "2) 0~1 점수 (높을수록 매수)\n"
        "3) 판단 이유 (2~3문장)",
        "",
        "답변 형식 (JSON): {\"recommendation\": \"매수|보류|매도\", \"score\": 0.0~1.0, \"reason\": \"...\"}",
    ]
    prompt = "\n".join(prompt_lines)

    payload = {
        "model": "abab6.5s-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300,
    }
    headers = {
        "Authorization": f"Bearer {config.MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = httpx.post(LLM_ENDPOINT, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # Lightweight JSON parse — strip markdown fences if present
        import json, re
        content = re.sub(r"```json|```", "", content).strip()
        result = json.loads(content)
        return {
            "recommendation": result.get("recommendation", "보류"),
            "score": float(result.get("score", 0.5)),
            "reason": result.get("reason", ""),
        }
    except Exception as e:
        return {"recommendation": "보류", "score": 0.5,
                "reason": f"LLM 호출 실패: {e}"}


def score_from_recommendation(rec: str) -> float:
    """Map recommendation string to normalised score 0~1."""
    return LLM_SCORES.get(rec, 0.5)
