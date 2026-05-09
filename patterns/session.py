"""
PatternTrader - Session & Auction Patterns (no look-ahead)
6 signals: SessionSweepBull, SessionSweepBear,
ExhaustionBull, ExhaustionBear,
CompressionBreakBull, CompressionBreakBear

Session = price sweeps session high/low then reverses.
Exhaustion = big move on declining volume.
Compression = tight range then explosive break.
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils


class SessionSweepBull(PatternDetector):
    """Price sweeps below the session low (rolling 24-bar low)
    then closes back above = institutional reversal."""
    name = "SessionSweepBull"
    category = "session"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        # Use 24-bar rolling low as proxy for session low
        session_low = df["low"].rolling(24).min().shift(1)
        low = df["low"]
        close = df["close"]
        sweep = (low < session_low) & (close > session_low) & session_low.notna()
        sig[sweep & utils.is_bullish(df)] = 1
        return sig


class SessionSweepBear(PatternDetector):
    """Price sweeps above session high then reverses down."""
    name = "SessionSweepBear"
    category = "session"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        session_high = df["high"].rolling(24).max().shift(1)
        high = df["high"]
        close = df["close"]
        sweep = (high > session_high) & (close < session_high) & session_high.notna()
        sig[sweep & utils.is_bearish(df)] = -1
        return sig


class ExhaustionBull(PatternDetector):
    """Strong bearish move (> 1.5x ATR down) on declining volume
    = sellers exhausted, expect reversal up."""
    name = "ExhaustionBull"
    category = "session"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        atr_val = utils.atr(df)
        close = df["close"]
        move_down = df["open"] - close  # positive when bearish
        vol = df["volume"]
        vol_declining = (vol < vol.shift(1)) & (vol.shift(1) < vol.shift(2))

        big_bearish = move_down > 1.5 * atr_val
        sig[big_bearish & vol_declining] = 1
        return sig


class ExhaustionBear(PatternDetector):
    """Strong bullish move on declining volume = buyers exhausted."""
    name = "ExhaustionBear"
    category = "session"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        atr_val = utils.atr(df)
        close = df["close"]
        move_up = close - df["open"]
        vol = df["volume"]
        vol_declining = (vol < vol.shift(1)) & (vol.shift(1) < vol.shift(2))

        big_bullish = move_up > 1.5 * atr_val
        sig[big_bullish & vol_declining] = -1
        return sig


class CompressionBreakBull(PatternDetector):
    """Volatility compression (ATR < 60% of 50-bar avg) followed by
    bullish expansion (close > 10-bar high). Squeeze -> breakout."""
    name = "CompressionBreakBull"
    category = "session"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        atr_val = utils.atr(df)
        avg_atr = atr_val.rolling(50).mean()

        # Was compressed in the last few bars
        compressed = atr_val.rolling(5).mean() < avg_atr * 0.6
        prev_compressed = compressed.shift(1)

        # Now breaking out
        close = df["close"]
        high_10 = df["high"].rolling(10).max().shift(1)
        breakout = close > high_10

        sig[prev_compressed & breakout & utils.is_bullish(df)] = 1
        return sig


class CompressionBreakBear(PatternDetector):
    """Compression then bearish breakout below 10-bar low."""
    name = "CompressionBreakBear"
    category = "session"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        atr_val = utils.atr(df)
        avg_atr = atr_val.rolling(50).mean()

        compressed = atr_val.rolling(5).mean() < avg_atr * 0.6
        prev_compressed = compressed.shift(1)

        close = df["close"]
        low_10 = df["low"].rolling(10).min().shift(1)
        breakdown = close < low_10

        sig[prev_compressed & breakdown & utils.is_bearish(df)] = -1
        return sig


ALL_SESSION = [
    SessionSweepBull, SessionSweepBear,
    ExhaustionBull, ExhaustionBear,
    CompressionBreakBull, CompressionBreakBear,
]
