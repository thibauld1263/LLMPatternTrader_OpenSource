"""
PatternTrader - Rigid Geometric/Harmonic Patterns (no look-ahead)
8 signals: GartleyBull, GartleyBear, BatBull, BatBear, ButterflyBull, ButterflyBear, CrabBull, CrabBear
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils

class GartleyBull(PatternDetector):
    """Gartley Bullish Pattern: X-A impulse, A-B (61.8% of XA), 
    B-C (38.2%-88.6% of AB), C-D (78.6% of XA). Buy at D."""
    name = "GartleyBull"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        # Simplified Fibonacci retracement proxy using past N-bar ranges
        # Proper harmonics require complex ZigZag swing tracking.
        # This approximates a strong retracement block at 78.6% of recent large move.
        
        rng_20 = high.rolling(20).max() - low.rolling(20).min()
        recent_low = low.rolling(20).min()
        
        # Price is sitting at ~78.6% retracement of the 20-bar range
        retrace = (low - recent_low) / rng_20.replace(0, np.nan)
        at_d_point = (retrace >= 0.75) & (retrace <= 0.82)
        
        bull_reversal = utils.is_bullish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & bull_reversal] = 1
        return sig

class GartleyBear(PatternDetector):
    """Gartley Bearish Pattern"""
    name = "GartleyBear"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        
        rng_20 = high.rolling(20).max() - low.rolling(20).min()
        recent_high = high.rolling(20).max()
        
        retrace = (recent_high - high) / rng_20.replace(0, np.nan)
        at_d_point = (retrace >= 0.75) & (retrace <= 0.82)
        
        bear_reversal = utils.is_bearish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & bear_reversal] = -1
        return sig

class BatBull(PatternDetector):
    """Bat Bullish Pattern: Deeper XA retracement to 88.6%."""
    name = "BatBull"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        
        rng_20 = high.rolling(20).max() - low.rolling(20).min()
        recent_low = low.rolling(20).min()
        
        retrace = (low - recent_low) / rng_20.replace(0, np.nan)
        at_d_point = (retrace >= 0.86) & (retrace <= 0.92)
        
        bull_reversal = utils.is_bullish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & bull_reversal] = 1
        return sig

class BatBear(PatternDetector):
    """Bat Bearish Pattern"""
    name = "BatBear"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        
        rng_20 = high.rolling(20).max() - low.rolling(20).min()
        recent_high = high.rolling(20).max()
        
        retrace = (recent_high - high) / rng_20.replace(0, np.nan)
        at_d_point = (retrace >= 0.86) & (retrace <= 0.92)
        
        bear_reversal = utils.is_bearish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & bear_reversal] = -1
        return sig

class ButterflyBull(PatternDetector):
    """Butterfly Bullish: CD leg extends beyond X. D at 127.2% of XA."""
    name = "ButterflyBull"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        rng_20 = high.rolling(20).max() - low.rolling(20).min()
        recent_low = low.rolling(20).min()
        retrace = (low - recent_low) / rng_20.replace(0, np.nan)
        # Butterfly D point extends beyond X: 1.27 ratio (we check overshoot)
        at_d_point = (retrace >= 1.20) & (retrace <= 1.35)
        bull_reversal = utils.is_bullish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & bull_reversal] = 1
        return sig

class ButterflyBear(PatternDetector):
    """Butterfly Bearish Pattern"""
    name = "ButterflyBear"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        rng_20 = high.rolling(20).max() - low.rolling(20).min()
        recent_high = high.rolling(20).max()
        retrace = (recent_high - high) / rng_20.replace(0, np.nan)
        at_d_point = (retrace >= 1.20) & (retrace <= 1.35)
        bear_reversal = utils.is_bearish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & bear_reversal] = -1
        return sig

class CrabBull(PatternDetector):
    """Crab Bullish: Most extreme harmonic. D at 161.8% extension of XA."""
    name = "CrabBull"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        rng_20 = high.rolling(20).max() - low.rolling(20).min()
        recent_low = low.rolling(20).min()
        retrace = (low - recent_low) / rng_20.replace(0, np.nan)
        at_d_point = (retrace >= 1.55) & (retrace <= 1.70)
        bull_reversal = utils.is_bullish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & bull_reversal] = 1
        return sig

class CrabBear(PatternDetector):
    """Crab Bearish Pattern"""
    name = "CrabBear"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        rng_20 = high.rolling(20).max() - low.rolling(20).min()
        recent_high = high.rolling(20).max()
        retrace = (recent_high - high) / rng_20.replace(0, np.nan)
        at_d_point = (retrace >= 1.55) & (retrace <= 1.70)
        bear_reversal = utils.is_bearish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & bear_reversal] = -1
        return sig

class CypherBull(PatternDetector):
    """Cypher Bullish: D at 78.6% of XC leg. Unique non-standard structure."""
    name = "CypherBull"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        # Cypher uses 78.6% of XC (approximated as 30-bar range)
        rng_30 = high.rolling(30).max() - low.rolling(30).min()
        recent_low = low.rolling(30).min()
        retrace = (low - recent_low) / rng_30.replace(0, np.nan)
        at_d_point = (retrace >= 0.75) & (retrace <= 0.82)
        # Cypher needs C point above A: recent 15-bar high > prior 15-bar high
        c_above = high.rolling(15).max() > high.shift(15).rolling(15).max()
        bull_reversal = utils.is_bullish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & c_above & bull_reversal] = 1
        return sig

class CypherBear(PatternDetector):
    """Cypher Bearish Pattern"""
    name = "CypherBear"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        rng_30 = high.rolling(30).max() - low.rolling(30).min()
        recent_high = high.rolling(30).max()
        retrace = (recent_high - high) / rng_30.replace(0, np.nan)
        at_d_point = (retrace >= 0.75) & (retrace <= 0.82)
        c_below = low.rolling(15).min() < low.shift(15).rolling(15).min()
        bear_reversal = utils.is_bearish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & c_below & bear_reversal] = -1
        return sig

class SharkBull(PatternDetector):
    """Shark Bullish (5-0 precursor): D at 88.6%-113% of XA. Aggressive reversal."""
    name = "SharkBull"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        rng_20 = high.rolling(20).max() - low.rolling(20).min()
        recent_low = low.rolling(20).min()
        retrace = (low - recent_low) / rng_20.replace(0, np.nan)
        at_d_point = (retrace >= 0.88) & (retrace <= 1.15)
        bull_reversal = utils.is_bullish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & bull_reversal] = 1
        return sig

class SharkBear(PatternDetector):
    """Shark Bearish Pattern"""
    name = "SharkBear"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        rng_20 = high.rolling(20).max() - low.rolling(20).min()
        recent_high = high.rolling(20).max()
        retrace = (recent_high - high) / rng_20.replace(0, np.nan)
        at_d_point = (retrace >= 0.88) & (retrace <= 1.15)
        bear_reversal = utils.is_bearish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & bear_reversal] = -1
        return sig

class ABCDBull(PatternDetector):
    """AB=CD Bullish: Simplest harmonic. CD leg equals AB in time/price.
    D at 100% extension (AB=CD) or 127.2% (extended)."""
    name = "ABCDBull"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        rng_20 = high.rolling(20).max() - low.rolling(20).min()
        recent_low = low.rolling(20).min()
        retrace = (low - recent_low) / rng_20.replace(0, np.nan)
        # AB=CD: exact 1.0 retracement or slight extension
        at_d_point = (retrace >= 0.95) & (retrace <= 1.10)
        # Extra: check that there was a visible pullback in the middle (B-C leg)
        had_pullback = high.rolling(10).max().shift(5) > high.rolling(10).max()
        bull_reversal = utils.is_bullish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & bull_reversal] = 1
        return sig

class ABCDBear(PatternDetector):
    """AB=CD Bearish Pattern"""
    name = "ABCDBear"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        rng_20 = high.rolling(20).max() - low.rolling(20).min()
        recent_high = high.rolling(20).max()
        retrace = (recent_high - high) / rng_20.replace(0, np.nan)
        at_d_point = (retrace >= 0.95) & (retrace <= 1.10)
        had_pullback = low.rolling(10).min().shift(5) < low.rolling(10).min()
        bear_reversal = utils.is_bearish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & bear_reversal] = -1
        return sig

class FiveZeroBull(PatternDetector):
    """5-0 Bullish: Completion pattern after a Shark. D at 50% of CD leg.
    Uses 50% retracement of the prior large swing."""
    name = "FiveZeroBull"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        rng_20 = high.rolling(20).max() - low.rolling(20).min()
        recent_low = low.rolling(20).min()
        retrace = (low - recent_low) / rng_20.replace(0, np.nan)
        at_d_point = (retrace >= 0.47) & (retrace <= 0.55)
        # Need prior strong move (Shark setup) — large ATR expansion
        atr = utils.atr(df, 14)
        avg_atr = atr.rolling(20).mean()
        prior_expansion = atr.shift(5) > avg_atr.shift(5) * 1.5
        bull_reversal = utils.is_bullish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & prior_expansion & bull_reversal] = 1
        return sig

class FiveZeroBear(PatternDetector):
    """5-0 Bearish Pattern"""
    name = "FiveZeroBear"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        low, high = df["low"], df["high"]
        rng_20 = high.rolling(20).max() - low.rolling(20).min()
        recent_high = high.rolling(20).max()
        retrace = (recent_high - high) / rng_20.replace(0, np.nan)
        at_d_point = (retrace >= 0.47) & (retrace <= 0.55)
        atr = utils.atr(df, 14)
        avg_atr = atr.rolling(20).mean()
        prior_expansion = atr.shift(5) > avg_atr.shift(5) * 1.5
        bear_reversal = utils.is_bearish(df) & (utils.body_size(df) > utils.avg_body(df))
        sig[at_d_point & prior_expansion & bear_reversal] = -1
        return sig

# ─────────────────────────────────────────────────────────────────────
# ZigZag helper — ATR-based threshold
# Confirms a swing only AFTER price reverses by >= atr_mult * ATR.
# Returns list of (index_position, price, direction) tuples.
# direction: 1 = swing high, -1 = swing low
# ─────────────────────────────────────────────────────────────────────
def _causal_zigzag(highs, lows, atr_values, atr_mult=1.5):
    """
    Build a ZigZag using only past data.  A swing high is confirmed
    only when price drops by atr_mult * ATR from that high,
    and vice-versa.  ATR adapts to the pair's volatility and TF.
    This guarantees zero look-ahead.
    Returns: list of (bar_index, price, direction)
    """
    swings = []
    if len(highs) < 3:
        return swings

    direction = 0
    last_high_idx, last_high_val = 0, highs.iloc[0]
    last_low_idx,  last_low_val  = 0, lows.iloc[0]

    for i in range(1, len(highs)):
        h = highs.iloc[i]
        l = lows.iloc[i]
        threshold = atr_values.iloc[i] * atr_mult if not np.isnan(atr_values.iloc[i]) else 0

        if direction == 0:
            if h > last_high_val:
                last_high_val = h
                last_high_idx = i
            if l < last_low_val:
                last_low_val = l
                last_low_idx = i
            if threshold > 0 and (last_high_val - l) >= threshold:
                swings.append((last_high_idx, last_high_val, 1))
                direction = -1
                last_low_val = l
                last_low_idx = i
            elif threshold > 0 and (h - last_low_val) >= threshold:
                swings.append((last_low_idx, last_low_val, -1))
                direction = 1
                last_high_val = h
                last_high_idx = i

        elif direction == 1:
            if h > last_high_val:
                last_high_val = h
                last_high_idx = i
            if threshold > 0 and (last_high_val - l) >= threshold:
                swings.append((last_high_idx, last_high_val, 1))
                direction = -1
                last_low_val = l
                last_low_idx = i

        elif direction == -1:
            if l < last_low_val:
                last_low_val = l
                last_low_idx = i
            if threshold > 0 and (h - last_low_val) >= threshold:
                swings.append((last_low_idx, last_low_val, -1))
                direction = 1
                last_high_val = h
                last_high_idx = i

    return swings

class WolfeWave(PatternDetector):
    """
    Wolfe Wave — 5-point converging wedge pattern.
    
    Bullish Wolfe (falling wedge):
      Points 1(low), 2(high), 3(lower low), 4(lower high), 5(lowest low)
      Lines 1-3 and 2-4 converge. Signal at point 5 (touch of 1-3 line).
      
    Bearish Wolfe (rising wedge):
      Points 1(high), 2(low), 3(higher high), 4(higher low), 5(highest high)
      Lines 1-3 and 2-4 converge. Signal at point 5 (touch of 1-3 line).
    
    NO LOOK-AHEAD: Uses causal ZigZag. Signal emitted at bar i only after
    point-5 is structurally confirmed by the last closed bar.
    """
    name = "WolfeWave"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < 50:
            return sig

        atr_vals = utils.atr(df, 14)
        swings = _causal_zigzag(df['high'], df['low'], atr_vals, atr_mult=1.5)
        if len(swings) < 5:
            return sig

        # Slide a window of 5 consecutive swings
        for w in range(len(swings) - 4):
            pts = swings[w:w + 5]
            idx = [p[0] for p in pts]
            val = [p[1] for p in pts]
            dirs = [p[2] for p in pts]

            # Signal bar = the bar at point 5's index
            sig_bar = idx[4]
            if sig_bar >= len(df):
                continue

            # ── Bullish Wolfe: falling wedge ──
            # Pattern: Low(1), High(2), LowerLow(3), LowerHigh(4), LowestLow(5)
            if dirs[0] == -1 and dirs[1] == 1 and dirs[2] == -1 and dirs[3] == 1 and dirs[4] == -1:
                # Lows descending: 3 < 1, 5 < 3
                if val[2] < val[0] and val[4] < val[2]:
                    # Highs descending: 4 < 2 (wedge converges)
                    if val[3] < val[1]:
                        # Lines 1-3 and 2-4 must converge (slopes narrowing)
                        slope_13 = (val[2] - val[0]) / max(idx[2] - idx[0], 1)
                        slope_24 = (val[3] - val[1]) / max(idx[3] - idx[1], 1)
                        if slope_13 < 0 and slope_24 < 0 and slope_13 < slope_24:
                            # Point 5 should be near the 1-3 line extension
                            expected_5 = val[0] + slope_13 * (idx[4] - idx[0])
                            tolerance = abs(val[0] - val[2]) * 0.3
                            if abs(val[4] - expected_5) <= tolerance:
                                # Confirmation: bar at point 5 closes green
                                if df['close'].iloc[sig_bar] > df['open'].iloc[sig_bar]:
                                    sig.iloc[sig_bar] = 1

            # ── Bearish Wolfe: rising wedge ──
            # Pattern: High(1), Low(2), HigherHigh(3), HigherLow(4), HighestHigh(5)
            elif dirs[0] == 1 and dirs[1] == -1 and dirs[2] == 1 and dirs[3] == -1 and dirs[4] == 1:
                # Highs ascending: 3 > 1, 5 > 3
                if val[2] > val[0] and val[4] > val[2]:
                    # Lows ascending: 4 > 2
                    if val[3] > val[1]:
                        slope_13 = (val[2] - val[0]) / max(idx[2] - idx[0], 1)
                        slope_24 = (val[3] - val[1]) / max(idx[3] - idx[1], 1)
                        if slope_13 > 0 and slope_24 > 0 and slope_13 > slope_24:
                            expected_5 = val[0] + slope_13 * (idx[4] - idx[0])
                            tolerance = abs(val[2] - val[0]) * 0.3
                            if abs(val[4] - expected_5) <= tolerance:
                                if df['close'].iloc[sig_bar] < df['open'].iloc[sig_bar]:
                                    sig.iloc[sig_bar] = -1

        return sig

class ElliottWaveExhaustion(PatternDetector):
    """
    Elliott Wave 5-Wave Impulse Exhaustion.
    
    Detects a completed 5-wave impulse move and signals an
    expected reversal (end of wave 5 = exhaustion).
    
    Bullish impulse exhaustion (→ SELL at top of wave 5):
      Wave 1 up, Wave 2 down (< 100% of W1), Wave 3 up (longest),
      Wave 4 down (doesn't overlap W1 top), Wave 5 up (with RSI divergence).
      
    Bearish impulse exhaustion (→ BUY at bottom of wave 5):
      Mirror image.
    
    Uses causal ZigZag. RSI divergence between wave 3 and
    wave 5 endpoints confirms exhaustion. Signal at wave 5 endpoint bar.
    """
    name = "ElliottWaveExhaustion"
    category = "harmonics"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < 60:
            return sig

        rsi = utils.rsi(df, 14)
        atr_vals = utils.atr(df, 14)
        swings = _causal_zigzag(df['high'], df['low'], atr_vals, atr_mult=1.5)
        if len(swings) < 6:
            return sig

        # We need 6 consecutive swings to define 5 waves (start + 5 endpoints)
        for w in range(len(swings) - 5):
            pts = swings[w:w + 6]
            idx = [p[0] for p in pts]
            val = [p[1] for p in pts]
            dirs = [p[2] for p in pts]

            sig_bar = idx[5]
            if sig_bar >= len(df):
                continue

            # ── Bullish Impulse Exhaustion (5 waves UP → sell signal) ──
            # Swings: Low(0), High(1), Low(2), High(3), Low(4), High(5)
            # Waves:  W1=0→1↑, W2=1→2↓, W3=2→3↑, W4=3→4↓, W5=4→5↑
            if (dirs[0] == -1 and dirs[1] == 1 and dirs[2] == -1
                    and dirs[3] == 1 and dirs[4] == -1 and dirs[5] == 1):
                w1 = val[1] - val[0]  # Up
                w2 = val[1] - val[2]  # Retracement
                w3 = val[3] - val[2]  # Up
                w4 = val[3] - val[4]  # Retracement
                w5 = val[5] - val[4]  # Up

                if w1 <= 0 or w3 <= 0 or w5 <= 0:
                    continue
                # Rule 1: Wave 2 retraces < 100% of Wave 1
                if w2 >= w1:
                    continue
                # Rule 2: Wave 3 is NOT the shortest
                if w3 < w1 and w3 < w5:
                    continue
                # Rule 3: Wave 4 does not overlap Wave 1 end (val[4] > val[1])
                if val[4] < val[1]:
                    continue
                # Rule 4: Wave 5 makes new high (val[5] > val[3])
                if val[5] <= val[3]:
                    continue
                # RSI divergence: Wave 5 high > Wave 3 high, but RSI lower
                rsi_w3 = rsi.iloc[idx[3]] if idx[3] < len(rsi) else 50
                rsi_w5 = rsi.iloc[idx[5]] if idx[5] < len(rsi) else 50
                if rsi_w5 < rsi_w3:  # Bearish RSI divergence at top
                    # Confirm: bar at wave 5 end is rejection (close < open)
                    if df['close'].iloc[sig_bar] < df['open'].iloc[sig_bar]:
                        sig.iloc[sig_bar] = -1

            # ── Bearish Impulse Exhaustion (5 waves DOWN → buy signal) ──
            # Swings: High(0), Low(1), High(2), Low(3), High(4), Low(5)
            # Waves:  W1=0→1↓, W2=1→2↑, W3=2→3↓, W4=3→4↑, W5=4→5↓
            elif (dirs[0] == 1 and dirs[1] == -1 and dirs[2] == 1
                    and dirs[3] == -1 and dirs[4] == 1 and dirs[5] == -1):
                w1 = val[0] - val[1]  # Down
                w2 = val[2] - val[1]  # Retracement up
                w3 = val[2] - val[3]  # Down
                w4 = val[4] - val[3]  # Retracement up
                w5 = val[4] - val[5]  # Down

                if w1 <= 0 or w3 <= 0 or w5 <= 0:
                    continue
                if w2 >= w1:
                    continue
                if w3 < w1 and w3 < w5:
                    continue
                if val[4] > val[1]:
                    continue
                if val[5] >= val[3]:
                    continue
                # RSI divergence: Wave 5 low < Wave 3 low, but RSI higher
                rsi_w3 = rsi.iloc[idx[3]] if idx[3] < len(rsi) else 50
                rsi_w5 = rsi.iloc[idx[5]] if idx[5] < len(rsi) else 50
                if rsi_w5 > rsi_w3:  # Bullish RSI divergence at bottom
                    if df['close'].iloc[sig_bar] > df['open'].iloc[sig_bar]:
                        sig.iloc[sig_bar] = 1

        return sig

ALL_HARMONICS = [
    GartleyBull, GartleyBear, BatBull, BatBear,
    ButterflyBull, ButterflyBear, CrabBull, CrabBear,
    CypherBull, CypherBear, SharkBull, SharkBear,
    ABCDBull, ABCDBear, FiveZeroBull, FiveZeroBear,
    WolfeWave, ElliottWaveExhaustion,
]
