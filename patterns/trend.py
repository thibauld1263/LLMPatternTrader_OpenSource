"""
PatternTrader - Trend Patterns
6 patterns: EMA_Cross, SMA_Cross, TrendlineBreakUp, TrendlineBreakDown,
SupportBreak, ResistanceBreak
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils
import config

class EMACross(PatternDetector):
    name = "EMA_Cross"
    category = "trend"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        fast = utils.ema(df["close"], config.EMA_FAST)
        slow = utils.ema(df["close"], config.EMA_SLOW)
        prev_fast = fast.shift(1)
        prev_slow = slow.shift(1)
        sig[(prev_fast <= prev_slow) & (fast > slow)] = 1
        sig[(prev_fast >= prev_slow) & (fast < slow)] = -1
        return sig

class SMACross(PatternDetector):
    name = "SMA_Cross"
    category = "trend"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        fast = utils.sma(df["close"], config.EMA_FAST)
        slow = utils.sma(df["close"], config.EMA_SLOW)
        prev_fast = fast.shift(1)
        prev_slow = slow.shift(1)
        sig[(prev_fast <= prev_slow) & (fast > slow)] = 1
        sig[(prev_fast >= prev_slow) & (fast < slow)] = -1
        return sig

class TrendlineBreakUp(PatternDetector):
    name = "TrendlineBreakUp"
    category = "trend"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        close = df["close"]
        period = 20
        rolling_high = close.rolling(period).max()
        prev_close = close.shift(1)
        prev_rh = rolling_high.shift(1)
        sig[(prev_close <= prev_rh) & (close > rolling_high)] = 1
        return sig

class TrendlineBreakDown(PatternDetector):
    name = "TrendlineBreakDown"
    category = "trend"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        close = df["close"]
        period = 20
        rolling_low = close.rolling(period).min()
        prev_close = close.shift(1)
        prev_rl = rolling_low.shift(1)
        sig[(prev_close >= prev_rl) & (close < rolling_low)] = -1
        return sig

class SupportBreak(PatternDetector):
    name = "SupportBreak"
    category = "trend"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        support = utils.swing_low_values(df)
        close = df["close"]
        prev_close = close.shift(1)
        mask = (prev_close >= support.shift(1)) & (close < support) & support.notna()
        sig[mask] = -1
        return sig

class ResistanceBreak(PatternDetector):
    name = "ResistanceBreak"
    category = "trend"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        resistance = utils.swing_high_values(df)
        close = df["close"]
        prev_close = close.shift(1)
        mask = (prev_close <= resistance.shift(1)) & (close > resistance) & resistance.notna()
        sig[mask] = 1
        return sig

ALL_TREND = [
    EMACross, SMACross,
    TrendlineBreakUp, TrendlineBreakDown,
    SupportBreak, ResistanceBreak,
]
