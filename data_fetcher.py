"""
PatternTrader - Data Fetcher
MT5 data retrieval with pickle caching, multi-timeframe.
Now uses date-based fetching (from DATA_START_DATE) instead of fixed bar counts.
"""

import os
import pickle
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
from typing import Dict, Optional, Tuple

import config


class MT5DataFetcher:

    def __init__(self):
        self.initialized = False
        os.makedirs(config.CACHE_DIR, exist_ok=True)

    # ── Connection ────────────────────────────────────────────────────

    def initialize(self, path: Optional[str] = None) -> bool:
        if path and os.path.exists(path):
            ok = mt5.initialize(path=path)
        else:
            ok = mt5.initialize()
        if not ok:
            print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
            return False
        ver = mt5.version()
        print(f"[OK] MT5 Connected - build {ver[0]}")
        self.initialized = True
        return True

    def shutdown(self):
        if self.initialized:
            mt5.shutdown()
            print("[OK] MT5 Disconnected")
            self.initialized = False

    # ── Cache ─────────────────────────────────────────────────────────

    def _cache_path(self, symbol: str, tf_name: str) -> str:
        start_tag = config.DATA_START_DATE.strftime("%Y%m%d")
        return os.path.join(config.CACHE_DIR, f"{symbol}_{tf_name}_from{start_tag}.pkl")

    def _load_cache(self, symbol: str, tf_name: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(symbol, tf_name)
        if not os.path.exists(path):
            return None
        age = datetime.now().timestamp() - os.path.getmtime(path)
        if age > 86_400:
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cache(self, symbol: str, tf_name: str, df: pd.DataFrame):
        try:
            with open(self._cache_path(symbol, tf_name), "wb") as f:
                pickle.dump(df, f)
        except Exception as e:
            print(f"  [WARN] Cache save failed: {e}")

    # ── Fetch ─────────────────────────────────────────────────────────

    def fetch_symbol(self, symbol: str, tf_const: int, bars: int,
                     tf_name: str = "", use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Fetch by bar count — used for lightweight subsequent-run checks (e.g. 500 bars)."""
        if not self.initialized:
            print(f"  [ERROR] MT5 not initialized")
            return None
        if not mt5.symbol_select(symbol, True):
            print(f"  [WARN] Cannot select {symbol}")
            return None
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars)
        if rates is None or len(rates) == 0:
            print(f"  [WARN] No data for {symbol}")
            return None
        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("datetime", inplace=True)
        df = df[["open", "high", "low", "close", "tick_volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        return df

    def fetch_data(self, symbol: str, tf_name: str, use_cache: bool = True,
                   bars: int = 20000) -> Optional[pd.DataFrame]:
        """Fetch data by bar count — uses the dashboard's bars_to_fetch setting."""
        tf_cfg = config.TIMEFRAMES.get(tf_name)
        if not tf_cfg:
            return None

        if use_cache:
            cached = self._load_cache(symbol, tf_name)
            if cached is not None:
                return cached

        if not self.initialized:
            print(f"  [ERROR] MT5 not initialized")
            return None
        if not mt5.symbol_select(symbol, True):
            print(f"  [WARN] Cannot select {symbol}")
            return None

        rates = mt5.copy_rates_from_pos(symbol, tf_cfg["tf_const"], 0, bars)
        if rates is None or len(rates) == 0:
            print(f"  [WARN] No data for {symbol}")
            return None

        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("datetime", inplace=True)
        df = df[["open", "high", "low", "close", "tick_volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]

        if use_cache:
            self._save_cache(symbol, tf_name, df)
        return df

    def fetch_all_timeframes(self, use_cache: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
        result = {}
        for tf_name, tf_cfg in config.TIMEFRAMES.items():
            print(f"\n  [{tf_name}] Fetching {len(config.SYMBOLS)} symbols (from {config.DATA_START_DATE.date()})...")
            tf_data = {}
            for i, sym in enumerate(config.SYMBOLS, 1):
                df = self.fetch_data(sym, tf_name, use_cache)
                if df is not None and len(df) >= 200:
                    tf_data[sym] = df
                    print(f"    [{i}/{len(config.SYMBOLS)}] {sym:10s} {len(df):>6,} bars")
                else:
                    print(f"    [{i}/{len(config.SYMBOLS)}] {sym:10s} FAILED")
            result[tf_name] = tf_data
        return result

    # ── Pip info ──────────────────────────────────────────────────────

    @staticmethod
    def pip_info(symbol: str) -> Tuple[float, float]:
        """Get pip size from MT5 dynamically.
        Uses point * 10 universally — works for all asset classes."""
        try:
            info = mt5.symbol_info(symbol)
            if info is not None:
                pip_size = info.point * 10
                return pip_size, (1.0 / pip_size if pip_size > 0 else 1.0)
        except Exception:
            pass
        # Fallback when MT5 unavailable
        sym = symbol.upper()
        if "JPY" in sym:
            return 0.01, 100.0
        return 0.0001, 10_000.0
