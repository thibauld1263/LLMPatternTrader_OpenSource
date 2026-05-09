"""
PatternTrader - Liquidity Patterns (no look-ahead)
4 signals: StopHuntBull, StopHuntBear, LiqGrabBull, LiqGrabBear

These detect institutional liquidity grabs: price sweeps past a
confirmed swing level then reverses back, indicating stop hunting.
All swings are causal (confirmed at bar i+order).
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils

class StopHuntBull(PatternDetector):
    """Price wicks below a confirmed swing low then closes above it.
    = Stops were swept below support, then price reversed up."""
    name = "StopHuntBull"
    category = "liquidity"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        swing_low = utils.swing_low_values(df)  # causal, forward-filled
        low = df["low"]
        close = df["close"]
        # Wick below swing low but close back above
        sweep = (low < swing_low) & (close > swing_low) & swing_low.notna()
        sig[sweep] = 1
        return sig

class StopHuntBear(PatternDetector):
    """Price wicks above a confirmed swing high then closes below it.
    = Stops swept above resistance, price reversed down."""
    name = "StopHuntBear"
    category = "liquidity"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        swing_high = utils.swing_high_values(df)  # causal
        high = df["high"]
        close = df["close"]
        sweep = (high > swing_high) & (close < swing_high) & swing_high.notna()
        sig[sweep] = -1
        return sig

class LiqGrabBull(PatternDetector):
    """Aggressive liquidity grab: bar makes new 20-bar low but closes
    in upper 30% of its range = strong rejection / demand."""
    name = "LiqGrabBull"
    category = "liquidity"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low = df["low"]
        close = df["close"]
        high = df["high"]
        rng = high - low
        rolling_low = low.rolling(20).min().shift(1)
        new_low = low < rolling_low
        close_in_upper = (close - low) > rng * 0.7
        sig[new_low & close_in_upper & (rng > 0)] = 1
        return sig

class LiqGrabBear(PatternDetector):
    """Bar makes new 20-bar high but closes in lower 30% = rejection / supply."""
    name = "LiqGrabBear"
    category = "liquidity"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low = df["low"]
        close = df["close"]
        high = df["high"]
        rng = high - low
        rolling_high = high.rolling(20).max().shift(1)
        new_high = high > rolling_high
        close_in_lower = (high - close) > rng * 0.7
        sig[new_high & close_in_lower & (rng > 0)] = -1
        return sig

ALL_LIQUIDITY = [StopHuntBull, StopHuntBear, LiqGrabBull, LiqGrabBear]
