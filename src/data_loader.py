from __future__ import annotations
"""Data loader for stock data from Kiwoom API and Yahoo Finance fallback."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import httpx
import pandas as pd
import sqlite3
from datetime import datetime
from typing import Optional

from src import config

class DataLoader:
    """Loads and caches OHLCV data from Kiwoom REST API or Yahoo Finance."""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR

    # ------------------------------------------------------------------
    # SQLite cache helpers
    # ------------------------------------------------------------------
    def _get_cache_db(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / "stocks_cache.db"

    def _load_cached_today(self, symbol: str) -> Optional[pd.DataFrame]:
        """Return cached data for today if fresh (< 6h old), else None."""
        db_path = self._get_cache_db()
        if not db_path.exists():
            return None
        conn = sqlite3.connect(db_path)
        try:
            df = pd.read_sql(
                "SELECT * FROM stock_daily WHERE symbol=? ORDER BY date DESC LIMIT 1",
                conn, params=(symbol,),
            )
        finally:
            conn.close()
        if df.empty:
            return None
        cached_date = pd.to_datetime(df["date"].iloc[0])
        age = datetime.now() - cached_date.tz_localize(None)
        if age.total_seconds() < 6 * 3600:
            return df
        return None

    def _save_to_cache(self, symbol: str, df: pd.DataFrame):
        db_path = self._get_cache_db()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        df = df.copy()
        df["symbol"] = symbol
        df.to_sql("stock_daily", conn, if_exists="append", index=False)
        conn.close()

    # ------------------------------------------------------------------
    # Kiwoom REST API
    # ------------------------------------------------------------------
    @staticmethod
    def _kiwoom_headers() -> dict:
        return {
            "Content-Type": "application/json",
            "authorization": f"Bearer {config.KIWOOM_API_KEY}",
            "contestant": "aiagent",
        }

    async def _download_daily_from_kiwoom(
        self, symbol: str, end_date: str, limit: int = 900
    ) -> pd.DataFrame:
        """Download daily OHLCV from Kiwoom REST API."""
        payload = {
            "exchtgcd": "",
            "fid_input_iscd": symbol,
            "fid_group_ind": "0",
            "fid_dt_last_date": end_date,
            "fid_evt_from_date": "",
            "fid_evt_to_date": "",
            "fid_max_rec_cnt": limit,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://openapi.kiwoom.com/api/dostk/ths/GetTHSnsmartstddaybassiseeksshcode",
                headers=self._kiwoom_headers(),
                json=payload,
            )
        resp.raise_for_status()
        data = resp.json()

        rows = data.get("thssnmthsv_dl_inform1", {}).get("output1", [])
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        rename = {
            "stck_shrn_iscd": "symbol",
            "stck_prpr": "close",
            "prdy_vrss": "change",
            "prdy_ctrt": "change_pct",
            "stck_oprc": "open",
            "stck_hgpr": "high",
            "stck_lwpr": "low",
            "acml_vol": "volume",
            "stck_mrkt_tot_amt": "market_cap",
            "req_dt": "date",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        for col in ["close", "open", "high", "low", "volume", "change", "change_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            df = df.sort_values("date").reset_index(drop=True)

        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        return df

    # ------------------------------------------------------------------
    # Yahoo Finance fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _download_daily_from_yahoo(symbol: str, period: str = "2y") -> pd.DataFrame:
        """Download daily OHLCV from Yahoo Finance (yfinance)."""
        import yfinance as yf
        tk = yf.Ticker(symbol)
        hist = tk.history(period=period, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()
        hist.columns = [c.lower() for c in hist.columns]
        hist = hist.rename(columns={"date": "date", "open": "open", "high": "high",
                                    "low": "low", "close": "close", "volume": "volume"})
        hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
        hist = hist.sort_values("date").reset_index(drop=True)
        return hist[["date", "open", "high", "low", "close", "volume"]]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def get_daily(self, symbol: str, source: str = "kiwoom",
                        end_date: str = "") -> pd.DataFrame:
        """Download daily OHLCV for a stock.

        Parameters
        ----------
        symbol   : Stock code
        source   : "kiwoom" (default) or "yahoo"
        end_date : "YYYYMMDD" for Kiwoom; "" for Yahoo
        """
        if source == "kiwoom":
            if not end_date:
                end_date = datetime.now().strftime("%Y%m%d")
            cached = self._load_cached_today(symbol)
            if cached is not None:
                return cached
            try:
                df = await self._download_daily_from_kiwoom(symbol, end_date)
            except Exception:
                df = pd.DataFrame()
            if not df.empty:
                self._save_to_cache(symbol, df)
            return df
        else:
            return self._download_daily_from_yahoo(symbol)

    async def get_kospi_codes(self) -> list[str]:
        """Return list of KOSPI stock codes from Kiwoom."""
        payload = {"exchtgcd": "01", "fid_input_iscd": "", "fid_group_ind": "0"}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://openapi.kiwoom.com/api/dostk/ths/GetTHSnsmartstddaybassiseeksshcode",
                headers=self._kiwoom_headers(),
                json=payload,
            )
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("thssnmthsv_dl_inform1", {}).get("output1", [])
        codes = [r["stck_shrt_iscd"] for r in rows if "stck_shrt_iscd" in r]
        return codes

    async def load_all_stocks(self, days: int = 1200) -> pd.DataFrame:
        """Load daily data for all KOSPI stocks.

        Returns a single DataFrame with an extra 'symbol' column.
        """
        codes = await self.get_kospi_codes()
        results = []
        for code in codes:
            df = await self.get_daily(code, source="kiwoom")
            if not df.empty:
                df = df.copy()
                df["symbol"] = code
                results.append(df)
        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)