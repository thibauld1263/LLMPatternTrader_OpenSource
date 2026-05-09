"""
PatternTrader - Advanced Institutional Patterns (no look-ahead)
10 signals: SFP_Bull, SFP_Bear, DisplacementBull, DisplacementBear,
WyckoffSpring, WyckoffUpthrust, RejectionBlockBull, RejectionBlockBear,
BreakerBull, BreakerBear
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils

class SFPBull(PatternDetector):
    """Swing Failure Pattern Bull: price makes a new low below a
    confirmed swing low, but CLOSES above the swing low.
    = Failed attempt to break lower, shorts trapped."""
    name = "SFP_Bull"
    category = "institutional"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        swing_low = utils.swing_low_values(df)
        low = df["low"]
        close = df["close"]
        opn = df["open"]
        body = utils.body_size(df)
        ab = utils.avg_body(df)

        # New low wick below swing, close above, decent body
        mask = (
            (low < swing_low) &
            (close > swing_low) &
            (close > opn) &  # bullish close
            (body > ab * 0.5) &
            swing_low.notna()
        )
        sig[mask] = 1
        return sig

class SFPBear(PatternDetector):
    """Swing Failure Pattern Bear: new high above swing high but
    closes below = longs trapped."""
    name = "SFP_Bear"
    category = "institutional"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        swing_high = utils.swing_high_values(df)
        high = df["high"]
        close = df["close"]
        opn = df["open"]
        body = utils.body_size(df)
        ab = utils.avg_body(df)

        mask = (
            (high > swing_high) &
            (close < swing_high) &
            (close < opn) &
            (body > ab * 0.5) &
            swing_high.notna()
        )
        sig[mask] = -1
        return sig

class DisplacementBull(PatternDetector):
    """Displacement = 3 consecutive bullish candles where each body
    is > 1.5x average body. Shows strong institutional buying intent."""
    name = "DisplacementBull"
    category = "institutional"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        body = utils.body_size(df)
        ab = utils.avg_body(df, 20)
        bull = utils.is_bullish(df)
        big = body > ab * 1.5

        # 3 consecutive big bullish candles
        consec = big & bull & big.shift(1) & bull.shift(1) & big.shift(2) & bull.shift(2)
        sig[consec] = 1
        return sig

class DisplacementBear(PatternDetector):
    """3 consecutive big bearish candles = institutional selling."""
    name = "DisplacementBear"
    category = "institutional"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        body = utils.body_size(df)
        ab = utils.avg_body(df, 20)
        bear = utils.is_bearish(df)
        big = body > ab * 1.5

        consec = big & bear & big.shift(1) & bear.shift(1) & big.shift(2) & bear.shift(2)
        sig[consec] = -1
        return sig

class WyckoffSpring(PatternDetector):
    """Wyckoff Spring: price dips below a consolidation low (20-bar)
    on LOW volume, then closes back inside = accumulation spring.
    Low volume = no real selling conviction = trap."""
    name = "WyckoffSpring"
    category = "institutional"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low = df["low"]
        close = df["close"]
        vol = df["volume"]
        avg_vol = vol.rolling(20).mean()
        range_low = low.rolling(20).min().shift(1)

        # Dip below range on low volume, close back above
        spring = (
            (low < range_low) &
            (close > range_low) &
            (vol < avg_vol * 0.8) &
            utils.is_bullish(df)
        )
        sig[spring] = 1
        return sig

class WyckoffUpthrust(PatternDetector):
    """Wyckoff Upthrust: price pokes above consolidation high on
    low volume then closes back below = distribution."""
    name = "WyckoffUpthrust"
    category = "institutional"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        high = df["high"]
        close = df["close"]
        vol = df["volume"]
        avg_vol = vol.rolling(20).mean()
        range_high = high.rolling(20).max().shift(1)

        upthrust = (
            (high > range_high) &
            (close < range_high) &
            (vol < avg_vol * 0.8) &
            utils.is_bearish(df)
        )
        sig[upthrust] = -1
        return sig

class RejectionBlockBull(PatternDetector):
    """Bullish Rejection Block: large lower wick (>60% of range)
    at a swing low zone. The wick area becomes a demand zone.
    Signal fires on the rejection candle itself."""
    name = "RejectionBlockBull"
    category = "institutional"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        ls = utils.lower_shadow(df)
        rng = utils.candle_range(df)
        body = utils.body_size(df)
        swing_low = utils.swing_low_values(df)
        atr_val = utils.atr(df)

        # Long lower wick at swing low zone
        big_wick = ls > rng * 0.6
        small_body = body < rng * 0.3
        at_support = (df["low"] - swing_low).abs() < atr_val * 0.5

        sig[big_wick & small_body & at_support & (rng > 0) & swing_low.notna()] = 1
        return sig

class RejectionBlockBear(PatternDetector):
    """Bearish Rejection Block: large upper wick at swing high zone."""
    name = "RejectionBlockBear"
    category = "institutional"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        us = utils.upper_shadow(df)
        rng = utils.candle_range(df)
        body = utils.body_size(df)
        swing_high = utils.swing_high_values(df)
        atr_val = utils.atr(df)

        big_wick = us > rng * 0.6
        small_body = body < rng * 0.3
        at_resistance = (df["high"] - swing_high).abs() < atr_val * 0.5

        sig[big_wick & small_body & at_resistance & (rng > 0) & swing_high.notna()] = -1
        return sig

class BreakerBull(PatternDetector):
    """Breaker Block Bull: an old bearish order block that gets 
    broken to the upside. The zone flips from supply to demand.
    When price returns to this zone = buy."""
    name = "BreakerBull"
    category = "institutional"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        o, c, high, low = df["open"], df["close"], df["high"], df["low"]
        atr_val = utils.atr(df)

        # Track bearish OBs that get broken
        ob_high = pd.Series(np.nan, index=df.index)
        ob_low = pd.Series(np.nan, index=df.index)
        breaker_high = pd.Series(np.nan, index=df.index)
        breaker_low = pd.Series(np.nan, index=df.index)

        for i in range(3, len(df)):
            # Detect bearish OB: last bullish candle before bearish impulse
            bearish_impulse = (o.iloc[i-1] - c.iloc[i-1]) > 2 * atr_val.iloc[i-1]
            if bearish_impulse and c.iloc[i-2] > o.iloc[i-2]:
                ob_high.iloc[i] = c.iloc[i-2]
                ob_low.iloc[i] = o.iloc[i-2]

        ob_high_ff = ob_high.ffill()
        ob_low_ff = ob_low.ffill()

        # Check if OB gets broken (close above OB high) -> becomes breaker
        broken = (c > ob_high_ff) & ob_high_ff.notna()
        for i in range(len(df)):
            if broken.iloc[i]:
                breaker_high.iloc[i] = ob_high_ff.iloc[i]
                breaker_low.iloc[i] = ob_low_ff.iloc[i]

        bk_high = breaker_high.ffill()
        bk_low = breaker_low.ffill()

        # Signal when price returns to breaker zone
        touches = (low <= bk_high) & (c > bk_low) & bk_high.notna()
        first = touches & ~touches.shift(1).fillna(False).astype(bool)
        sig[first] = 1
        return sig

class BreakerBear(PatternDetector):
    """Breaker Block Bear: old bullish OB broken to downside, flips to supply."""
    name = "BreakerBear"
    category = "institutional"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        o, c, high, low = df["open"], df["close"], df["high"], df["low"]
        atr_val = utils.atr(df)

        ob_high = pd.Series(np.nan, index=df.index)
        ob_low = pd.Series(np.nan, index=df.index)
        breaker_high = pd.Series(np.nan, index=df.index)
        breaker_low = pd.Series(np.nan, index=df.index)

        for i in range(3, len(df)):
            bullish_impulse = (c.iloc[i-1] - o.iloc[i-1]) > 2 * atr_val.iloc[i-1]
            if bullish_impulse and c.iloc[i-2] < o.iloc[i-2]:
                ob_high.iloc[i] = o.iloc[i-2]
                ob_low.iloc[i] = c.iloc[i-2]

        ob_high_ff = ob_high.ffill()
        ob_low_ff = ob_low.ffill()

        broken = (c < ob_low_ff) & ob_low_ff.notna()
        for i in range(len(df)):
            if broken.iloc[i]:
                breaker_high.iloc[i] = ob_high_ff.iloc[i]
                breaker_low.iloc[i] = ob_low_ff.iloc[i]

        bk_high = breaker_high.ffill()
        bk_low = breaker_low.ffill()

        touches = (high >= bk_low) & (c < bk_high) & bk_low.notna()
        first = touches & ~touches.shift(1).fillna(False).astype(bool)
        sig[first] = -1
        return sig

ALL_INSTITUTIONAL = [
    SFPBull, SFPBear,
    DisplacementBull, DisplacementBear,
    WyckoffSpring, WyckoffUpthrust,
    RejectionBlockBull, RejectionBlockBear,
    BreakerBull, BreakerBear,
]
