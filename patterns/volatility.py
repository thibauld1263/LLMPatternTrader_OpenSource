"""
PatternTrader - Volatility Patterns
4 patterns: BB_BreakUp, BB_BreakDown, BB_Squeeze, ATR_Expansion
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils
import config


class BBBreakUp(PatternDetector):
    name = "BB_BreakUp"
    category = "volatility"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        upper, _, _ = utils.bollinger(df)
        close = df["close"]
        prev_close = close.shift(1)
        sig[(prev_close <= upper.shift(1)) & (close > upper)] = 1
        return sig


class BBBreakDown(PatternDetector):
    name = "BB_BreakDown"
    category = "volatility"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        _, _, lower = utils.bollinger(df)
        close = df["close"]
        prev_close = close.shift(1)
        sig[(prev_close >= lower.shift(1)) & (close < lower)] = -1
        return sig


class BBSqueeze(PatternDetector):
    name = "BB_Squeeze"
    category = "volatility"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        upper, _, lower = utils.bollinger(df)
        bw = (upper - lower) / ((upper + lower) / 2)
        avg_bw = bw.rolling(50).mean()
        squeeze = bw < avg_bw * 0.5
        expansion = ~squeeze
        prev_squeeze = squeeze.shift(1)
        close = df["close"]
        ema20 = utils.ema(close, 20)
        sig[prev_squeeze & expansion & (close > ema20)] = 1
        sig[prev_squeeze & expansion & (close < ema20)] = -1
        return sig


class ATRExpansion(PatternDetector):
    name = "ATR_Expansion"
    category = "volatility"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        atr_val = utils.atr(df)
        avg_atr = atr_val.rolling(50).mean()
        expansion = atr_val > avg_atr * 1.5
        prev_low = atr_val.shift(1) <= avg_atr.shift(1) * 1.5
        close = df["close"]
        ema20 = utils.ema(close, 20)
        sig[expansion & prev_low & (close > ema20)] = 1
        sig[expansion & prev_low & (close < ema20)] = -1
        return sig


ALL_VOLATILITY = [
    BBBreakUp, BBBreakDown, BBSqueeze, ATRExpansion,
]
