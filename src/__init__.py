from __future__ import annotations
"""LSTMScreener - LSTM Stock Screening + MiniMax LLM Hybrid Analysis."""
from src.screener import Screener

# Backward-compatible wrappers (old function-based API)
async def screen_all(model_path: str = "", top_n: int = 50):
    s = Screener(model_path=model_path)
    return await s.screen_all(top_n=top_n)

async def analyze_top(n: int = 20, model_path: str = ""):
    s = Screener(model_path=model_path)
    return await s.analyze_top(n=n)

__all__ = ["Screener", "screen_all", "analyze_top"]