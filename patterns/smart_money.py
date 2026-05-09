"""
PatternTrader - Smart Money Structure (no look-ahead)
6 signals: BOS_Bull, BOS_Bear, CHoCH_Bull, CHoCH_Bear,
FailedBreakBull, FailedBreakBear

BOS = Break of Structure (confirmed swing break with momentum).
CHoCH = Change of Character (first break against prevailing trend).
FailedBreak = false breakout that traps traders.
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils


class BOSBull(PatternDetector):
    """Break of Structure Bullish: close breaks above the most recent
    confirmed swing high with a strong candle (body > avg)."""
    name = "BOS_Bull"
    category = "smart_money"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        swing_high = utils.swing_high_values(df)  # causal
        close = df["close"]
        prev_close = close.shift(1)
        body = utils.body_size(df)
        ab = utils.avg_body(df)

        # Break above swing high with conviction
        breaks = (
            (prev_close <= swing_high.shift(1)) &
            (close > swing_high) &
            (body > ab * 0.8) &
            utils.is_bullish(df) &
            swing_high.notna()
        )
        sig[breaks] = 1
        return sig


class BOSBear(PatternDetector):
    """Break of Structure Bearish: close breaks below swing low with conviction."""
    name = "BOS_Bear"
    category = "smart_money"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        swing_low = utils.swing_low_values(df)
        close = df["close"]
        prev_close = close.shift(1)
        body = utils.body_size(df)
        ab = utils.avg_body(df)

        breaks = (
            (prev_close >= swing_low.shift(1)) &
            (close < swing_low) &
            (body > ab * 0.8) &
            utils.is_bearish(df) &
            swing_low.notna()
        )
        sig[breaks] = -1
        return sig


class CHoCHBull(PatternDetector):
    """Change of Character Bullish: price was making lower lows,
    then breaks above the most recent lower high.
    = First sign downtrend is ending."""
    name = "CHoCH_Bull"
    category = "smart_money"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Detect downtrend: 3 consecutive lower lows over rolling 15 bars
        period = 15
        lows_now = low.rolling(period).min()
        lows_prev = low.shift(period).rolling(period).min()
        in_downtrend = lows_now < lows_prev

        # Recent high as resistance
        recent_high = high.rolling(period).max().shift(1)

        # Break above recent high while in downtrend
        breaks = in_downtrend & (close > recent_high) & utils.is_bullish(df)
        sig[breaks] = 1
        return sig


class CHoCHBear(PatternDetector):
    """Change of Character Bearish: price was making higher highs,
    then breaks below the most recent higher low."""
    name = "CHoCH_Bear"
    category = "smart_money"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        period = 15
        highs_now = high.rolling(period).max()
        highs_prev = high.shift(period).rolling(period).max()
        in_uptrend = highs_now > highs_prev

        recent_low = low.rolling(period).min().shift(1)

        breaks = in_uptrend & (close < recent_low) & utils.is_bearish(df)
        sig[breaks] = -1
        return sig


class FailedBreakBull(PatternDetector):
    """Failed breakout below support: price breaks below swing low
    then closes back above it within 3 bars = trapped shorts."""
    name = "FailedBreakBull"
    category = "smart_money"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        swing_low = utils.swing_low_values(df)
        close = df["close"]
        low = df["low"]

        # Track if we broke below swing low recently (within 3 bars)
        broke_below = pd.Series(False, index=df.index)
        for shift in range(1, 4):
            broke_below = broke_below | (
                (low.shift(shift) < swing_low.shift(shift)) &
                swing_low.shift(shift).notna()
            )

        # Now closing back above = failed break
        back_above = (close > swing_low) & swing_low.notna()
        sig[broke_below & back_above & utils.is_bullish(df)] = 1
        return sig


class FailedBreakBear(PatternDetector):
    """Failed breakout above resistance: price breaks above swing high
    then closes back below = trapped longs."""
    name = "FailedBreakBear"
    category = "smart_money"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        swing_high = utils.swing_high_values(df)
        close = df["close"]
        high = df["high"]

        broke_above = pd.Series(False, index=df.index)
        for shift in range(1, 4):
            broke_above = broke_above | (
                (high.shift(shift) > swing_high.shift(shift)) &
                swing_high.shift(shift).notna()
            )

        back_below = (close < swing_high) & swing_high.notna()
        sig[broke_above & back_below & utils.is_bearish(df)] = -1
        return sig


ALL_SMART_MONEY = [
    BOSBull, BOSBear,
    CHoCHBull, CHoCHBear,
    FailedBreakBull, FailedBreakBear,
]
