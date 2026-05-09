"""
PatternTrader - Advanced Structural Patterns (no look-ahead)
12 signals: FVGReturnBull, FVGReturnBear, QuasimodoBull, QuasimodoBear,
CompressionBull, CompressionBear, ThreeDriveBull, ThreeDriveBear,
ReturnToOriginBull, ReturnToOriginBear, HikkakeBull, HikkakeBear
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils

class FVGReturnBull(PatternDetector):
    """Fair Value Gap Return Bull: price aggressively breaks up leaving
    an imbalance (FVG), then slowly drifts down to retest the gap."""
    name = "FVG_ReturnBull"
    category = "structural"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high, close = df["low"], df["high"], df["close"]
        
        # Track bullish FVGs (low of candle 3 > high of candle 1)
        fvg_high = pd.Series(np.nan, index=df.index)
        fvg_low = pd.Series(np.nan, index=df.index)
        
        for i in range(2, len(df)):
            if low.iloc[i] > high.iloc[i-2]:
                fvg_high.iloc[i] = low.iloc[i]
                fvg_low.iloc[i] = high.iloc[i-2]
                
        fvg_high_ff = fvg_high.ffill()
        fvg_low_ff = fvg_low.ffill()
        
        # Price subsequently dips into the FVG zone to fill it
        filled = (low <= fvg_high_ff) & (close >= fvg_low_ff) & fvg_high_ff.notna()
        first_fill = filled & ~filled.shift(1).fillna(False).astype(bool)
        sig[first_fill & utils.is_bullish(df)] = 1
        return sig

class FVGReturnBear(PatternDetector):
    """Fair Value Gap Return Bear: FVG left by strong drop, later retested."""
    name = "FVG_ReturnBear"
    category = "structural"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high, close = df["low"], df["high"], df["close"]
        
        # Track bearish FVGs (high of candle 3 < low of candle 1)
        fvg_high = pd.Series(np.nan, index=df.index)
        fvg_low = pd.Series(np.nan, index=df.index)
        
        for i in range(2, len(df)):
            if high.iloc[i] < low.iloc[i-2]:
                fvg_high.iloc[i] = low.iloc[i-2]
                fvg_low.iloc[i] = high.iloc[i]
                
        fvg_high_ff = fvg_high.ffill()
        fvg_low_ff = fvg_low.ffill()
        
        filled = (high >= fvg_low_ff) & (close <= fvg_high_ff) & fvg_high_ff.notna()
        first_fill = filled & ~filled.shift(1).fillna(False).astype(bool)
        sig[first_fill & utils.is_bearish(df)] = -1
        return sig

class QuasimodoBull(PatternDetector):
    """Quasimodo (Over and Under) Buy: High, Low, Higher High (sweep stops), 
    Lower Low (break structure). Entry is on return to the first Left Shoulder low."""
    name = "QuasimodoBull"
    category = "structural"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        # Proper QML requires tracking multiple swings in order. 
        # This is a simplified causal approximation using recent swings and ATR.
        swing_h = utils.swing_high_values(df)
        swing_l = utils.swing_low_values(df)
        low = df["low"]
        close = df["close"]
        atr = utils.atr(df)
        
        # Assuming we recently broke structural low (LL), and we are now testing an old key low
        for i in range(20, len(df)):
            # If current price touches a past significant swing low that was broken
            if swing_l.iloc[i] > 0: # Causal confirmation of swing low
                recent_swing_l = swing_l.iloc[i]
                near_shoulder = (low.iloc[i] - recent_swing_l).abs() < atr.iloc[i] * 0.5
                if near_shoulder and close.iloc[i] > low.iloc[i]:
                    sig.iloc[i] = 1
        return sig

class QuasimodoBear(PatternDetector):
    """Quasimodo Bear (Over and Under) Sell."""
    name = "QuasimodoBear"
    category = "structural"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        swing_h = utils.swing_high_values(df)
        high = df["high"]
        close = df["close"]
        atr = utils.atr(df)
        
        for i in range(20, len(df)):
            if swing_h.iloc[i] > 0:
                recent_swing_h = swing_h.iloc[i]
                near_shoulder = (high.iloc[i] - recent_swing_h).abs() < atr.iloc[i] * 0.5
                if near_shoulder and close.iloc[i] < high.iloc[i]:
                    sig.iloc[i] = -1
        return sig

class CompressionBull(PatternDetector):
    """Compression to Support: series of shrinking candles descending into support, 
    followed by a strong bullish breakout/rejection."""
    name = "CompressionBull"
    category = "structural"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        rng = utils.candle_range(df)
        rng_5 = rng.rolling(5).sum()
        rng_prev_5 = rng.shift(5).rolling(5).sum()
        
        swing_l = utils.swing_low_values(df)
        atr_val = utils.atr(df)
        
        # The last 5 candles are significantly smaller than the 5 before them
        compressing = rng_5 < (rng_prev_5 * 0.6)
        at_support = (df["low"] - swing_l).abs() < atr_val * 0.5
        breakout = utils.body_size(df) > utils.avg_body(df) * 1.5
        
        sig[compressing & at_support & breakout & utils.is_bullish(df)] = 1
        return sig

class CompressionBear(PatternDetector):
    """Compression to Resistance: shrinking candles ascending to resistance, 
    then strong rejection down."""
    name = "CompressionBear"
    category = "structural"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        rng = utils.candle_range(df)
        rng_5 = rng.rolling(5).sum()
        rng_prev_5 = rng.shift(5).rolling(5).sum()
        
        swing_h = utils.swing_high_values(df)
        atr_val = utils.atr(df)
        
        compressing = rng_5 < (rng_prev_5 * 0.6)
        at_resistance = (df["high"] - swing_h).abs() < atr_val * 0.5
        breakout = utils.body_size(df) > utils.avg_body(df) * 1.5
        
        sig[compressing & at_resistance & breakout & utils.is_bearish(df)] = -1
        return sig

class ThreeDriveBull(PatternDetector):
    """Three-Drive Bull: 3 consecutive lower lows, but RSI makes higher lows (Divergence)."""
    name = "ThreeDriveBull"
    category = "structural"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low = df["low"]
        rsi_val = utils.rsi(df)
        
        # Simplified structural check (every ~10 bars finding mins)
        l1 = low.rolling(10).min()
        l2 = l1.shift(15)
        l3 = l2.shift(15)
        
        r1 = rsi_val.rolling(10).min()
        r2 = r1.shift(15)
        r3 = r2.shift(15)
        
        price_dives = (l1 < l2) & (l2 < l3)
        rsi_diverges = (r1 > r2) & (r2 > r3)
        
        strong_close = utils.is_bullish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[price_dives & rsi_diverges & strong_close] = 1
        return sig

class ThreeDriveBear(PatternDetector):
    """Three-Drive Bear: 3 consecutive higher highs, but RSI makes lower highs."""
    name = "ThreeDriveBear"
    category = "structural"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        high = df["high"]
        rsi_val = utils.rsi(df)
        
        h1 = high.rolling(10).max()
        h2 = h1.shift(15)
        h3 = h2.shift(15)
        
        r1 = rsi_val.rolling(10).max()
        r2 = r1.shift(15)
        r3 = r2.shift(15)
        
        price_push = (h1 > h2) & (h2 > h3)
        rsi_diverges = (r1 < r2) & (r2 < r3)
        
        strong_close = utils.is_bearish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[price_push & rsi_diverges & strong_close] = -1
        return sig

class ReturnToOriginBull(PatternDetector):
    """Return to Origin (RTO) Bull: first retest of the genesis candle of a massive impulse."""
    name = "ReturnToOriginBull"
    category = "structural"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        atr = utils.atr(df)
        o, c, l = df["open"], df["close"], df["low"]
        
        origin_low = pd.Series(np.nan, index=df.index)
        for i in range(5, len(df)):
            # Massive impulse check
            impulse = (c.iloc[i-1] - o.iloc[i-1]) > (5 * atr.iloc[i-1])
            if impulse:
                # The low of the candle that started it
                origin_low.iloc[i] = l.iloc[i-1]
                
        origin_low_ff = origin_low.ffill()
        
        touches = (l <= origin_low_ff + atr * 0.3) & origin_low_ff.notna()
        first_touch = touches & ~touches.shift(1).fillna(False).astype(bool)
        sig[first_touch & utils.is_bullish(df)] = 1
        return sig

class ReturnToOriginBear(PatternDetector):
    """Return to Origin (RTO) Bear."""
    name = "ReturnToOriginBear"
    category = "structural"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        atr = utils.atr(df)
        o, c, h = df["open"], df["close"], df["high"]
        
        origin_high = pd.Series(np.nan, index=df.index)
        for i in range(5, len(df)):
            impulse = (o.iloc[i-1] - c.iloc[i-1]) > (5 * atr.iloc[i-1])
            if impulse:
                origin_high.iloc[i] = h.iloc[i-1]
                
        origin_high_ff = origin_high.ffill()
        
        touches = (h >= origin_high_ff - atr * 0.3) & origin_high_ff.notna()
        first_touch = touches & ~touches.shift(1).fillna(False).astype(bool)
        sig[first_touch & utils.is_bearish(df)] = -1
        return sig

class HikkakeBull(PatternDetector):
    """Hikkake Bull: an inside bar is broken to the downside, but within 3 bars 
    it strongly reverses to break the inside bar's high."""
    name = "HikkakeBull"
    category = "structural"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        h, l, c = df["high"], df["low"], df["close"]
        
        # Inside bar definition: H < prev_H and L > prev_L
        inside_bar = (h < h.shift(1)) & (l > l.shift(1))
        
        # Shift 1, 2, 3 looking for a previous inside bar that broke low then broke high
        ib_highs = h.shift(1)[inside_bar.shift(1).fillna(False).astype(bool)]
        
        # Need loop to correlate specific broken inside bars. Simplified logic:
        prev_h = h.shift(1)
        prev_l = l.shift(1)
        prev_ib = (prev_h < h.shift(2)) & (prev_l > l.shift(2))
        
        # Breakout to the downside
        false_breakout = (l < prev_l) & prev_ib
        # Next bar breaks back above the inside bar high
        reversal = (c > h.shift(2)) & false_breakout.shift(1)
        
        sig[reversal] = 1
        return sig

class HikkakeBear(PatternDetector):
    """Hikkake Bear: inside bar broken upside, strongly reverses to break low."""
    name = "HikkakeBear"
    category = "structural"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        h, l, c = df["high"], df["low"], df["close"]
        
        prev_h = h.shift(1)
        prev_l = l.shift(1)
        prev_ib = (prev_h < h.shift(2)) & (prev_l > l.shift(2))
        
        false_breakout = (h > prev_h) & prev_ib
        reversal = (c < l.shift(2)) & false_breakout.shift(1)
        
        sig[reversal] = -1
        return sig

ALL_STRUCTURAL = [
    FVGReturnBull, FVGReturnBear,
    QuasimodoBull, QuasimodoBear,
    CompressionBull, CompressionBear,
    ThreeDriveBull, ThreeDriveBear,
    ReturnToOriginBull, ReturnToOriginBear,
    HikkakeBull, HikkakeBear
]
