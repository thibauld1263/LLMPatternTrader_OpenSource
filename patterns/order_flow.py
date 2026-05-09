"""
PatternTrader - Order Flow Patterns (no look-ahead)
6 signals: OrderBlockBull, OrderBlockBear, FVG_Bull, FVG_Bear,
AbsorptionBull, AbsorptionBear

Order blocks = last opposing candle before a strong impulse.
FVG = imbalance gap between candle 1 and candle 3.
Absorption = high volume + small body = institutional accumulation.
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils

class OrderBlockBull(PatternDetector):
    """Last bearish candle before a strong bullish impulse (>2x ATR).
    When price returns to that candle's zone = buy signal.
    Signal fires on the return, not on the impulse."""
    name = "OrderBlockBull"
    category = "order_flow"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        o, c, high, low = df["open"], df["close"], df["high"], df["low"]
        atr_val = utils.atr(df)
        body = utils.body_size(df)

        ob_high = pd.Series(np.nan, index=df.index)
        ob_low = pd.Series(np.nan, index=df.index)

        for i in range(2, len(df)):
            # Check if bar i-1 was a strong bullish impulse
            impulse = (c.iloc[i-1] - o.iloc[i-1]) > 2 * atr_val.iloc[i-1]
            if not impulse:
                continue
            # Find last bearish candle before the impulse
            if c.iloc[i-2] < o.iloc[i-2]:
                ob_high.iloc[i-1] = o.iloc[i-2]  # top of OB
                ob_low.iloc[i-1] = c.iloc[i-2]   # bottom of OB

        # Forward-fill OB zones
        ob_high = ob_high.ffill()
        ob_low = ob_low.ffill()

        # Signal when price returns to OB zone
        touches = (low <= ob_high) & (c > ob_low) & ob_high.notna()
        # Only signal once per zone (use shift to avoid re-triggering)
        first_touch = touches & ~touches.shift(1).fillna(False).astype(bool)
        sig[first_touch] = 1
        return sig

class OrderBlockBear(PatternDetector):
    """Last bullish candle before a strong bearish impulse.
    Signal fires when price returns to that zone."""
    name = "OrderBlockBear"
    category = "order_flow"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        o, c, high, low = df["open"], df["close"], df["high"], df["low"]
        atr_val = utils.atr(df)

        ob_high = pd.Series(np.nan, index=df.index)
        ob_low = pd.Series(np.nan, index=df.index)

        for i in range(2, len(df)):
            impulse = (o.iloc[i-1] - c.iloc[i-1]) > 2 * atr_val.iloc[i-1]
            if not impulse:
                continue
            if c.iloc[i-2] > o.iloc[i-2]:
                ob_high.iloc[i-1] = c.iloc[i-2]
                ob_low.iloc[i-1] = o.iloc[i-2]

        ob_high = ob_high.ffill()
        ob_low = ob_low.ffill()

        touches = (high >= ob_low) & (c < ob_high) & ob_low.notna()
        first_touch = touches & ~touches.shift(1).fillna(False).astype(bool)
        sig[first_touch] = -1
        return sig

class FVGBull(PatternDetector):
    """Bullish Fair Value Gap: candle[i-2].high < candle[i].low.
    Gap between candle 1 and 3 = imbalance.
    Signal fires when price later drops into the FVG zone."""
    name = "FVG_Bull"
    category = "order_flow"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        high = df["high"]
        low = df["low"]
        close = df["close"]

        fvg_top = pd.Series(np.nan, index=df.index)
        fvg_bot = pd.Series(np.nan, index=df.index)

        for i in range(2, len(df)):
            # Bullish FVG: bar[i]'s low > bar[i-2]'s high AND bar[i-1] is bullish
            if low.iloc[i] > high.iloc[i-2] and close.iloc[i-1] > df["open"].iloc[i-1]:
                fvg_top.iloc[i] = low.iloc[i]
                fvg_bot.iloc[i] = high.iloc[i-2]

        fvg_top = fvg_top.ffill()
        fvg_bot = fvg_bot.ffill()

        # Price enters the FVG from above
        enters = (low <= fvg_top) & (close > fvg_bot) & fvg_top.notna()
        first = enters & ~enters.shift(1).fillna(False).astype(bool)
        sig[first] = 1
        return sig

class FVGBear(PatternDetector):
    """Bearish FVG: candle[i-2].low > candle[i].high.
    Signal fires when price rises into the FVG zone."""
    name = "FVG_Bear"
    category = "order_flow"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        high = df["high"]
        low = df["low"]
        close = df["close"]

        fvg_top = pd.Series(np.nan, index=df.index)
        fvg_bot = pd.Series(np.nan, index=df.index)

        for i in range(2, len(df)):
            if high.iloc[i] < low.iloc[i-2] and close.iloc[i-1] < df["open"].iloc[i-1]:
                fvg_top.iloc[i] = low.iloc[i-2]
                fvg_bot.iloc[i] = high.iloc[i]

        fvg_top = fvg_top.ffill()
        fvg_bot = fvg_bot.ffill()

        enters = (high >= fvg_bot) & (close < fvg_top) & fvg_bot.notna()
        first = enters & ~enters.shift(1).fillna(False).astype(bool)
        sig[first] = -1
        return sig

class AbsorptionBull(PatternDetector):
    """High volume + small body at support = institutional buying.
    Volume > 2.5x avg, body < 25% of range, at/near swing low."""
    name = "AbsorptionBull"
    category = "order_flow"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        vol = df["volume"]
        avg_vol = vol.rolling(20).mean()
        body = utils.body_size(df)
        rng = utils.candle_range(df)
        swing_low = utils.swing_low_values(df)
        atr_val = utils.atr(df)

        high_vol = vol > avg_vol * 2.5
        small_body = body < rng * 0.25
        near_support = (df["low"] - swing_low).abs() < atr_val * 0.5

        sig[high_vol & small_body & near_support & (rng > 0) & swing_low.notna()] = 1
        return sig

class AbsorptionBear(PatternDetector):
    """High volume + small body at resistance = institutional selling."""
    name = "AbsorptionBear"
    category = "order_flow"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        vol = df["volume"]
        avg_vol = vol.rolling(20).mean()
        body = utils.body_size(df)
        rng = utils.candle_range(df)
        swing_high = utils.swing_high_values(df)
        atr_val = utils.atr(df)

        high_vol = vol > avg_vol * 2.5
        small_body = body < rng * 0.25
        near_resistance = (df["high"] - swing_high).abs() < atr_val * 0.5

        sig[high_vol & small_body & near_resistance & (rng > 0) & swing_high.notna()] = -1
        return sig

ALL_ORDER_FLOW = [
    OrderBlockBull, OrderBlockBear,
    FVGBull, FVGBear,
    AbsorptionBull, AbsorptionBear,
]
