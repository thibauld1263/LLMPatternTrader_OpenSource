"""
PatternTrader - Advanced Patterns 2
10 more patterns: TurtleSoupBull, TurtleSoupBear, TwoB_Bull, TwoB_Bear,
VolumeClimaxBull, VolumeClimaxBear, EqualLowsBull, EqualHighsBear,
EngulfingAtSupportBull, EngulfingAtResistanceBear
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils


class TurtleSoupBull(PatternDetector):
    """Linda Raschke's Turtle Soup: price breaks below 20-bar low
    by < 1 ATR, then closes back above = false breakout, buy."""
    name = "TurtleSoupBull"
    category = "advanced"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low = df["low"]
        close = df["close"]
        atr_val = utils.atr(df)
        rolling_low = low.rolling(20).min().shift(1)

        # Break below by small amount, close back above
        break_below = (low < rolling_low) & ((rolling_low - low) < atr_val)
        close_above = close > rolling_low
        sig[break_below & close_above & utils.is_bullish(df)] = 1
        return sig


class TurtleSoupBear(PatternDetector):
    """Turtle Soup Short: false breakout above 20-bar high."""
    name = "TurtleSoupBear"
    category = "advanced"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        high = df["high"]
        close = df["close"]
        atr_val = utils.atr(df)
        rolling_high = high.rolling(20).max().shift(1)

        break_above = (high > rolling_high) & ((high - rolling_high) < atr_val)
        close_below = close < rolling_high
        sig[break_above & close_below & utils.is_bearish(df)] = -1
        return sig


class TwoBBull(PatternDetector):
    """2B Reversal (Victor Sperandeo): price makes new low below
    prior swing low, fails to sustain, closes above prior low.
    Stronger than simple SFP because it requires prior test."""
    name = "2B_Bull"
    category = "advanced"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low = df["low"]
        close = df["close"]
        # Two tests of similar level
        prev_low_5 = low.rolling(5).min().shift(5)
        prev_low_10 = low.rolling(5).min().shift(10)
        atr_val = utils.atr(df)

        # Current low near or below prior lows (within 0.5 ATR)
        near_prior = (low - prev_low_5).abs() < atr_val * 0.5
        new_test = low <= prev_low_5
        # But closes well above
        strong_close = (close - low) > atr_val * 0.5
        sig[new_test & near_prior & strong_close & utils.is_bullish(df)] = 1
        return sig


class TwoBBear(PatternDetector):
    """2B Reversal Short: failed retest of swing high."""
    name = "2B_Bear"
    category = "advanced"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        high = df["high"]
        close = df["close"]
        prev_high_5 = high.rolling(5).max().shift(5)
        atr_val = utils.atr(df)

        near_prior = (high - prev_high_5).abs() < atr_val * 0.5
        new_test = high >= prev_high_5
        strong_close = (high - close) > atr_val * 0.5
        sig[new_test & near_prior & strong_close & utils.is_bearish(df)] = -1
        return sig


class VolumeClimaxBull(PatternDetector):
    """Volume Climax Buy: highest volume in 30 bars + bullish close
    near high of bar = institutional buying climax."""
    name = "VolClimaxBull"
    category = "advanced"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        vol = df["volume"]
        max_vol = vol.rolling(30).max()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        rng = high - low

        climax = vol >= max_vol
        bullish = utils.is_bullish(df)
        close_near_high = (close - low) > rng * 0.7
        sig[climax & bullish & close_near_high & (rng > 0)] = 1
        return sig


class VolumeClimaxBear(PatternDetector):
    """Volume Climax Sell: highest volume + bearish close near low."""
    name = "VolClimaxBear"
    category = "advanced"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        vol = df["volume"]
        max_vol = vol.rolling(30).max()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        rng = high - low

        climax = vol >= max_vol
        bearish = utils.is_bearish(df)
        close_near_low = (high - close) > rng * 0.7
        sig[climax & bearish & close_near_low & (rng > 0)] = -1
        return sig


class EqualLowsBull(PatternDetector):
    """Equal Lows (ICT): two or more lows within 0.2 ATR of each other
    over 10-30 bars = liquidity pool below. When price sweeps below
    then reverses = buy (stops were collected)."""
    name = "EqualLowsBull"
    category = "advanced"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low = df["low"]
        close = df["close"]
        atr_val = utils.atr(df)

        for i in range(30, len(df)):
            # Look for equal lows in last 10-30 bars
            window_lows = low.iloc[i-30:i]
            min_low = window_lows.min()
            # Count how many bars had lows near the minimum
            near = (window_lows - min_low).abs() < atr_val.iloc[i] * 0.2
            if near.sum() >= 2:
                # Current bar sweeps below and reverses
                if low.iloc[i] < min_low and close.iloc[i] > min_low:
                    sig.iloc[i] = 1
        return sig


class EqualHighsBear(PatternDetector):
    """Equal Highs: liquidity pool above equal highs, swept then reversed."""
    name = "EqualHighsBear"
    category = "advanced"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        high = df["high"]
        close = df["close"]
        atr_val = utils.atr(df)

        for i in range(30, len(df)):
            window_highs = high.iloc[i-30:i]
            max_high = window_highs.max()
            near = (max_high - window_highs).abs() < atr_val.iloc[i] * 0.2
            if near.sum() >= 2:
                if high.iloc[i] > max_high and close.iloc[i] < max_high:
                    sig.iloc[i] = -1
        return sig


class EngulfingAtSupport(PatternDetector):
    """Bullish engulfing specifically at a swing low zone
    = much higher probability than random engulfing."""
    name = "EngulfAtSupport"
    category = "advanced"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        o, c = df["open"], df["close"]
        prev_o, prev_c = o.shift(1), c.shift(1)
        swing_low = utils.swing_low_values(df)
        atr_val = utils.atr(df)

        engulf = (
            (prev_c < prev_o) &  # prev bearish
            (c > o) &             # current bullish
            (o <= prev_c) &       # open below prev close
            (c >= prev_o)         # close above prev open
        )
        at_support = (df["low"] - swing_low).abs() < atr_val * 0.5
        sig[engulf & at_support & swing_low.notna()] = 1
        return sig


class EngulfingAtResistance(PatternDetector):
    """Bearish engulfing at swing high zone."""
    name = "EngulfAtResist"
    category = "advanced"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        o, c = df["open"], df["close"]
        prev_o, prev_c = o.shift(1), c.shift(1)
        swing_high = utils.swing_high_values(df)
        atr_val = utils.atr(df)

        engulf = (
            (prev_c > prev_o) &
            (c < o) &
            (o >= prev_c) &
            (c <= prev_o)
        )
        at_resist = (df["high"] - swing_high).abs() < atr_val * 0.5
        sig[engulf & at_resist & swing_high.notna()] = -1
        return sig


ALL_ADVANCED = [
    TurtleSoupBull, TurtleSoupBear,
    TwoBBull, TwoBBear,
    VolumeClimaxBull, VolumeClimaxBear,
    EqualLowsBull, EqualHighsBear,
    EngulfingAtSupport, EngulfingAtResistance,
]
