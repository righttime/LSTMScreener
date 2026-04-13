from __future__ import annotations
"""Main screener: hybrid LSTM + LLM scoring."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Optional

from src import config
from src.data_loader import DataLoader
from src.feature_eng import FeatureEngineer
from src.lstm_model import LSTMModel
from src import llm_analyzer


class Screener:
    """Hybrid LSTM + LLM stock screener."""

    def __init__(self, config_path: str = ".env", model_path: str = ""):
        self.loader    = DataLoader()
        self.engineer  = FeatureEngineer(seq_len=config.SEQ_LEN)
        self.model     = LSTMModel(input_size=17, device="cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path) if model_path else Path(config.MODEL_DIR, "lstm_best.pt")

        # Load pretrained weights if available
        if self.model_path.exists():
            try:
                self.model.load(str(self.model_path))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # LSTM scoring helpers
    # ------------------------------------------------------------------
    async def _lstm_score_one(self, symbol: str) -> float:
        """Download, feature-engineer, and return LSTM probability."""
        df = await self.loader.get_daily(symbol, source="kiwoom")
        if df is None or len(df) < config.SEQ_LEN + 10:
            return 0.5
        df = self.engineer.add_indicators(df)
        feat = self.engineer.build_features(df)
        seq = feat.tail(config.SEQ_LEN).values
        if seq.shape[0] < config.SEQ_LEN:
            return 0.5
        # Normalize using the full feature df for scaler fitting
        scaled_all, _ = self.engineer.normalize_train(feat)
        scaler = self.engineer.scaler
        seq_scaled = scaler.transform(seq)
        X = torch.tensor(seq_scaled[np.newaxis, :], dtype=torch.float32).to(self.model.device)
        prob = self.model.predict_proba(X).item()
        return prob

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def screen_all(self, top_n: int = 50) -> List[Tuple[str, float]]:
        """Run LSTM prediction across all available stocks.

        Returns
        -------
        List of (symbol, lstm_prob) sorted descending, limited to top_n.
        """
        if not self.model.is_trained:
            # Return mock scores if no model yet
            return [("005930", 0.72), ("000660", 0.68), ("035720", 0.65)]

        codes = await self.loader.get_kospi_codes()
        tasks = [self._lstm_score_one(c) for c in codes]
        scores = await asyncio.gather(*tasks, return_exceptions=True)
        results = [(c, s) for c, s in zip(codes, scores) if isinstance(s, float)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    async def analyze_top(self, n: int = 20) -> List[dict]:
        """Hybrid screen + LLM analysis → final recommendations.

        Returns
        -------
        List of dicts with keys: symbol, company_name, lstm_prob, llm_score,
                                  final_score, recommendation, reason
        """
        lstm_results = await self.screen_all(top_n=n)
        final = []

        for symbol, lstm_prob in lstm_results:
            company_name = symbol  # placeholder
            llm_result = llm_analyzer.analyze_fundamental(
                ticker       = symbol,
                company_name = company_name,
                financials   = {},
                news         = [],
                sector_news  = "",
            )
            llm_score = llm_analyzer.score_from_recommendation(
                llm_result["recommendation"]
            )
            final_score = lstm_prob * config.LSTM_WEIGHT + llm_score * config.LLM_WEIGHT

            final.append({
                "symbol":        symbol,
                "company_name":  company_name,
                "lstm_prob":     round(lstm_prob, 4),
                "llm_score":     round(llm_score, 4),
                "final_score":   round(final_score, 4),
                "recommendation": llm_result["recommendation"],
                "reason":        llm_result["reason"],
            })

        final.sort(key=lambda x: x["final_score"], reverse=True)
        return final


if __name__ == "__main__":
    results = asyncio.run(Screener().screen_all(top_n=5))
    for r in results:
        print(r)