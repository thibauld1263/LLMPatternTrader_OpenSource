"""
PatternTrader - Complex Confluence Patterns
Advanced multi-factor patterns that combine structural, momentum, and volume analysis.
These are NOT redundant with existing patterns — each requires multiple conditions to align.
"""

import numpy as np
import pandas as pd
from patterns.base import PatternDetector
import ta_utils as utils


# ═══════════════════════════════════════════════════════════════════════
# 1. DIVERGENCE AT PIVOT — RSI divergence at a structural pivot level
# ═══════════════════════════════════════════════════════════════════════

class DivAtSupportBull(PatternDetector):
    """Bullish RSI divergence occurring near a macro daily Support Zone.
    
    Logic: price makes lower low + RSI makes higher low (classic bull div),
    AND current price is within 1 ATR of a Daily Support zone (via sr_zones clustering).
    This combines momentum exhaustion with real historically clustered structural 
    support rather than arbitrary mathematical formulas.
    """
    name = "DivAtSupportBull"
    category = "confluence"
    LOOKBACK = 30

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK + 50:
            return sig

        rsi_vals = utils.rsi(df, 14)
        atr_vals = utils.atr(df, 14)

        # Build efficient rolling Macro SR History
        # 1. Resample to Daily to find macro structures
        d_df = df.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        # 2. Reconstruct periodic zones to avoid lookahead bias
        sr_history = {}
        for i in range(50, len(d_df)):
            dt_end = d_df.index[i]
            # Use daily data only up to this point in history
            sr = utils.sr_zones(d_df.iloc[:i+1], lookback=100, num_zones=2)
            sr_history[dt_end] = sr

        if not sr_history:
            return sig

        # 3. Form a dataframe and shift by 1 day (so intraday periods use the PREVIOUS daily close zones)
        sr_df = pd.DataFrame.from_dict(sr_history, orient='index').shift(1)

        # 4. Forward fill these daily macro zones onto the intraday dataframe index
        sr_expanded = sr_df.reindex(df.index, method='ffill')
        s1 = sr_expanded.get('support_1', pd.Series(np.nan, index=df.index))
        s2 = sr_expanded.get('support_2', pd.Series(np.nan, index=df.index))

        for i in range(self.LOOKBACK, len(df)):
            atr_v = atr_vals.iloc[i]
            if pd.isna(atr_v) or atr_v <= 0:
                continue

            price = df['close'].iloc[i]
            s1_v = s1.iloc[i]
            s2_v = s2.iloc[i]
            if pd.isna(s1_v):
                continue

            # Check proximity to S1 or S2
            near_support = (abs(price - s1_v) < atr_v) or (not pd.isna(s2_v) and abs(price - s2_v) < atr_v)
            if not near_support:
                continue

            # Check RSI bullish divergence: price lower low + RSI higher low
            window = slice(i - self.LOOKBACK, i)
            price_window = df['low'].iloc[window]
            rsi_window = rsi_vals.iloc[window]

            # Find the lowest low in the window and check if current is lower
            prev_low_idx = price_window.idxmin()
            if prev_low_idx == df.index[i - 1]:
                continue
            prev_low_pos = df.index.get_loc(prev_low_idx)

            if df['low'].iloc[i] < df['low'].iloc[prev_low_pos]:
                # Price made lower low — check RSI made higher low
                if rsi_vals.iloc[i] > rsi_vals.iloc[prev_low_pos] and rsi_vals.iloc[i] < 40:
                    sig.iloc[i] = 1

        return sig


class DivAtResistanceBear(PatternDetector):
    """Bearish RSI divergence occurring near a macro daily Resistance Zone.
    
    Price makes higher high + RSI makes lower high, AND price is within 1 ATR
    of a Daily Resistance zone (via sr_zones clustering).
    """
    name = "DivAtResistanceBear"
    category = "confluence"
    LOOKBACK = 30

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK + 50:
            return sig

        rsi_vals = utils.rsi(df, 14)
        atr_vals = utils.atr(df, 14)

        # Build efficient rolling Macro SR History
        # 1. Resample to Daily to find macro structures
        d_df = df.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        # 2. Reconstruct periodic zones to avoid lookahead bias
        sr_history = {}
        for i in range(50, len(d_df)):
            dt_end = d_df.index[i]
            # Use daily data only up to this point in history
            sr = utils.sr_zones(d_df.iloc[:i+1], lookback=100, num_zones=2)
            sr_history[dt_end] = sr

        if not sr_history:
            return sig

        # 3. Form a dataframe and shift by 1 day
        sr_df = pd.DataFrame.from_dict(sr_history, orient='index').shift(1)

        # 4. Forward fill
        sr_expanded = sr_df.reindex(df.index, method='ffill')
        r1 = sr_expanded.get('resistance_1', pd.Series(np.nan, index=df.index))
        r2 = sr_expanded.get('resistance_2', pd.Series(np.nan, index=df.index))

        for i in range(self.LOOKBACK, len(df)):
            atr_v = atr_vals.iloc[i]
            if pd.isna(atr_v) or atr_v <= 0:
                continue

            price = df['close'].iloc[i]
            r1_v = r1.iloc[i]
            r2_v = r2.iloc[i]
            if pd.isna(r1_v):
                continue

            near_resist = (abs(price - r1_v) < atr_v) or (not pd.isna(r2_v) and abs(price - r2_v) < atr_v)
            if not near_resist:
                continue

            window = slice(i - self.LOOKBACK, i)
            prev_high_idx = df['high'].iloc[window].idxmax()
            if prev_high_idx == df.index[i - 1]:
                continue
            prev_high_pos = df.index.get_loc(prev_high_idx)

            if df['high'].iloc[i] > df['high'].iloc[prev_high_pos]:
                if rsi_vals.iloc[i] < rsi_vals.iloc[prev_high_pos] and rsi_vals.iloc[i] > 60:
                    sig.iloc[i] = -1

        return sig


# ═══════════════════════════════════════════════════════════════════════
# 2. DELTA DIVERGENCE — Buy/sell volume ratio diverging from price
# ═══════════════════════════════════════════════════════════════════════

class DeltaDivBull(PatternDetector):
    """Price makes lower lows but net buying volume (bull bar volume) is
    increasing relative to bear bar volume. Smart money is accumulating
    while price drops — reversal is brewing.
    
    Different from VolumePriceDivergence which just checks total vol vs price.
    This separates DIRECTIONAL volume flow.
    """
    name = "DeltaDivBull"
    category = "confluence"
    LOOKBACK = 20

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK + 10:
            return sig

        vol = df['volume']
        bull_bar = (df['close'] > df['open']).astype(float)
        bear_bar = (df['close'] < df['open']).astype(float)

        # Rolling buy/sell volume
        buy_vol = (vol * bull_bar).rolling(self.LOOKBACK).sum()
        sell_vol = (vol * bear_bar).rolling(self.LOOKBACK).sum()
        delta_ratio = buy_vol / sell_vol.replace(0, np.nan)

        # Price trend: lower lows over the lookback
        low_min = df['low'].rolling(self.LOOKBACK).min()

        for i in range(self.LOOKBACK + 5, len(df)):
            if pd.isna(delta_ratio.iloc[i]) or pd.isna(delta_ratio.iloc[i - 10]):
                continue

            # Price making lower lows (current low near rolling min)
            price_declining = df['low'].iloc[i] <= low_min.iloc[i] * 1.002

            # But buy/sell ratio is increasing (delta divergence)
            delta_rising = delta_ratio.iloc[i] > delta_ratio.iloc[i - 10] * 1.1

            # Delta shows buying dominance emerging
            if price_declining and delta_rising and delta_ratio.iloc[i] > 0.8:
                sig.iloc[i] = 1

        return sig


class DeltaDivBear(PatternDetector):
    """Price makes higher highs but net selling volume is increasing.
    Smart money is distributing while price rises.
    """
    name = "DeltaDivBear"
    category = "confluence"
    LOOKBACK = 20

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK + 10:
            return sig

        vol = df['volume']
        bull_bar = (df['close'] > df['open']).astype(float)
        bear_bar = (df['close'] < df['open']).astype(float)

        buy_vol = (vol * bull_bar).rolling(self.LOOKBACK).sum()
        sell_vol = (vol * bear_bar).rolling(self.LOOKBACK).sum()
        delta_ratio = sell_vol / buy_vol.replace(0, np.nan)

        high_max = df['high'].rolling(self.LOOKBACK).max()

        for i in range(self.LOOKBACK + 5, len(df)):
            if pd.isna(delta_ratio.iloc[i]) or pd.isna(delta_ratio.iloc[i - 10]):
                continue

            price_rising = df['high'].iloc[i] >= high_max.iloc[i] * 0.998
            delta_rising = delta_ratio.iloc[i] > delta_ratio.iloc[i - 10] * 1.1

            if price_rising and delta_rising and delta_ratio.iloc[i] > 0.8:
                sig.iloc[i] = -1

        return sig


# ═══════════════════════════════════════════════════════════════════════
# 3. WYCKOFF ACCUMULATION / DISTRIBUTION SEQUENCES
# ═══════════════════════════════════════════════════════════════════════

class WyckoffAccumSeq(PatternDetector):
    """Full Wyckoff accumulation sequence detection.
    
    Phases detected (within a 60-bar window):
    1. Selling Climax: volume spike (>2x avg) + large bearish bar + reversal
    2. Trading Range: at least 10 bars of sideways price action (ATR contracts)
    3. Spring: price dips below the range low then closes back inside
    4. Sign of Strength: bullish close above range midpoint with expanding volume
    
    Only fires when ALL phases complete in sequence.
    Different from WyckoffSpring which only detects phase 3 in isolation.
    """
    name = "WyckoffAccumSeq"
    category = "confluence"
    LOOKBACK = 80

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK:
            return sig

        vol = df['volume']
        avg_vol = vol.rolling(20).mean()
        atr_vals = utils.atr(df, 14)

        for i in range(self.LOOKBACK, len(df)):
            # PHASE 1: Find a selling climax in the lookback window
            climax_bar = None
            for j in range(i - self.LOOKBACK, i - 30):
                if pd.isna(avg_vol.iloc[j]) or avg_vol.iloc[j] <= 0:
                    continue
                is_climax = (
                    vol.iloc[j] > avg_vol.iloc[j] * 2.0 and
                    df['close'].iloc[j] < df['open'].iloc[j] and
                    (df['open'].iloc[j] - df['close'].iloc[j]) > atr_vals.iloc[j] * 1.0
                )
                if is_climax:
                    climax_bar = j
                    break

            if climax_bar is None:
                continue

            # PHASE 2: Trading range after climax (at least 15 bars of consolidation)
            range_start = climax_bar + 2
            range_end = min(range_start + 30, i - 5)
            if range_end - range_start < 15:
                continue

            range_slice = df.iloc[range_start:range_end]
            range_high = range_slice['high'].max()
            range_low = range_slice['low'].min()
            range_width = range_high - range_low

            if pd.isna(atr_vals.iloc[range_end]):
                continue
            # Range should be reasonably tight (less than 4 ATR wide)
            if range_width > atr_vals.iloc[range_end] * 4:
                continue

            # PHASE 3: Spring — price dips below range low then closes back inside
            spring_found = False
            for k in range(range_end, min(range_end + 15, i)):
                if df['low'].iloc[k] < range_low and df['close'].iloc[k] > range_low:
                    spring_found = True
                    break

            if not spring_found:
                continue

            # PHASE 4: Sign of Strength — close above range midpoint with vol
            range_mid = (range_high + range_low) / 2
            if (df['close'].iloc[i] > range_mid and
                df['close'].iloc[i] > df['open'].iloc[i] and
                vol.iloc[i] > avg_vol.iloc[i] * 1.2):
                sig.iloc[i] = 1

        return sig


class WyckoffDistribSeq(PatternDetector):
    """Full Wyckoff distribution sequence: Buying Climax → Range → Upthrust → 
    Sign of Weakness. Mirror of accumulation.
    """
    name = "WyckoffDistribSeq"
    category = "confluence"
    LOOKBACK = 80

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK:
            return sig

        vol = df['volume']
        avg_vol = vol.rolling(20).mean()
        atr_vals = utils.atr(df, 14)

        for i in range(self.LOOKBACK, len(df)):
            # PHASE 1: Buying climax
            climax_bar = None
            for j in range(i - self.LOOKBACK, i - 30):
                if pd.isna(avg_vol.iloc[j]) or avg_vol.iloc[j] <= 0:
                    continue
                is_climax = (
                    vol.iloc[j] > avg_vol.iloc[j] * 2.0 and
                    df['close'].iloc[j] > df['open'].iloc[j] and
                    (df['close'].iloc[j] - df['open'].iloc[j]) > atr_vals.iloc[j] * 1.0
                )
                if is_climax:
                    climax_bar = j
                    break

            if climax_bar is None:
                continue

            # PHASE 2: Trading range
            range_start = climax_bar + 2
            range_end = min(range_start + 30, i - 5)
            if range_end - range_start < 15:
                continue

            range_slice = df.iloc[range_start:range_end]
            range_high = range_slice['high'].max()
            range_low = range_slice['low'].min()
            range_width = range_high - range_low

            if pd.isna(atr_vals.iloc[range_end]):
                continue
            if range_width > atr_vals.iloc[range_end] * 4:
                continue

            # PHASE 3: Upthrust — price pokes above range high then closes inside
            upthrust_found = False
            for k in range(range_end, min(range_end + 15, i)):
                if df['high'].iloc[k] > range_high and df['close'].iloc[k] < range_high:
                    upthrust_found = True
                    break

            if not upthrust_found:
                continue

            # PHASE 4: Sign of Weakness
            range_mid = (range_high + range_low) / 2
            if (df['close'].iloc[i] < range_mid and
                df['close'].iloc[i] < df['open'].iloc[i] and
                vol.iloc[i] > avg_vol.iloc[i] * 1.2):
                sig.iloc[i] = -1

        return sig


# ═══════════════════════════════════════════════════════════════════════
# 4. FIBONACCI CLUSTER ZONES — Multi-swing fib convergence
# ═══════════════════════════════════════════════════════════════════════

class FibClusterBull(PatternDetector):
    """Detects when price enters a zone where multiple Fibonacci retracement
    levels from different swing moves converge (within 0.5 ATR).
    
    Uses the last 3 swing high→low moves, draws 38.2%, 50%, 61.8% for each,
    and checks if 3+ fib levels cluster together near current price.
    Fires a bullish signal when price touches a cluster from above (support).
    """
    name = "FibClusterBull"
    category = "confluence"
    LOOKBACK = 100
    FIB_LEVELS = [0.382, 0.5, 0.618]

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK:
            return sig

        swing_h = utils.swing_highs(df, 5)
        swing_l = utils.swing_lows(df, 5)
        atr_vals = utils.atr(df, 14)

        for i in range(self.LOOKBACK, len(df)):
            atr_v = atr_vals.iloc[i]
            if pd.isna(atr_v) or atr_v <= 0:
                continue

            # Collect recent swing highs and lows
            recent = df.iloc[max(0, i - self.LOOKBACK):i + 1]
            sh_mask = swing_h.iloc[max(0, i - self.LOOKBACK):i + 1]
            sl_mask = swing_l.iloc[max(0, i - self.LOOKBACK):i + 1]

            sh_indices = recent.index[sh_mask]
            sl_indices = recent.index[sl_mask]

            if len(sh_indices) < 2 or len(sl_indices) < 2:
                continue

            # Generate fib levels from last 3 swing high→low pairs
            fib_prices = []
            swings_used = 0
            for sh_idx in reversed(sh_indices):
                for sl_idx in reversed(sl_indices):
                    sh_pos = df.index.get_loc(sh_idx)
                    sl_pos = df.index.get_loc(sl_idx)
                    if sh_pos >= sl_pos:     # high must come before low for a down-swing
                        continue
                    h_val = df['high'].iloc[sh_pos]
                    l_val = df['low'].iloc[sl_pos]
                    swing_range = h_val - l_val
                    if swing_range < atr_v * 2:  # Minimum swing size
                        continue
                    for fib in self.FIB_LEVELS:
                        fib_prices.append(l_val + swing_range * fib)
                    swings_used += 1
                    if swings_used >= 3:
                        break
                if swings_used >= 3:
                    break

            if len(fib_prices) < 6:
                continue

            # Check for cluster: 3+ fib levels within 0.5 ATR of each other
            fib_prices.sort()
            price = df['close'].iloc[i]
            cluster_threshold = atr_v * 0.5

            for f_idx in range(len(fib_prices) - 2):
                cluster_center = fib_prices[f_idx]
                cluster_count = sum(
                    1 for fp in fib_prices
                    if abs(fp - cluster_center) < cluster_threshold
                )
                if cluster_count >= 3 and abs(price - cluster_center) < atr_v:
                    # Price is at the cluster zone — bullish bounce
                    if price <= cluster_center and df['close'].iloc[i] > df['open'].iloc[i]:
                        sig.iloc[i] = 1
                        break

        return sig


class FibClusterBear(PatternDetector):
    """Fibonacci cluster resistance — price enters a zone where multiple fib 
    levels converge, acting as resistance. Mirror of FibClusterBull.
    """
    name = "FibClusterBear"
    category = "confluence"
    LOOKBACK = 100
    FIB_LEVELS = [0.382, 0.5, 0.618]

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK:
            return sig

        swing_h = utils.swing_highs(df, 5)
        swing_l = utils.swing_lows(df, 5)
        atr_vals = utils.atr(df, 14)

        for i in range(self.LOOKBACK, len(df)):
            atr_v = atr_vals.iloc[i]
            if pd.isna(atr_v) or atr_v <= 0:
                continue

            recent = df.iloc[max(0, i - self.LOOKBACK):i + 1]
            sh_mask = swing_h.iloc[max(0, i - self.LOOKBACK):i + 1]
            sl_mask = swing_l.iloc[max(0, i - self.LOOKBACK):i + 1]

            sh_indices = recent.index[sh_mask]
            sl_indices = recent.index[sl_mask]

            if len(sh_indices) < 2 or len(sl_indices) < 2:
                continue

            # Fib levels from low→high swings (for resistance)
            fib_prices = []
            swings_used = 0
            for sl_idx in reversed(sl_indices):
                for sh_idx in reversed(sh_indices):
                    sl_pos = df.index.get_loc(sl_idx)
                    sh_pos = df.index.get_loc(sh_idx)
                    if sl_pos >= sh_pos:
                        continue
                    l_val = df['low'].iloc[sl_pos]
                    h_val = df['high'].iloc[sh_pos]
                    swing_range = h_val - l_val
                    if swing_range < atr_v * 2:
                        continue
                    for fib in self.FIB_LEVELS:
                        fib_prices.append(h_val - swing_range * fib)
                    swings_used += 1
                    if swings_used >= 3:
                        break
                if swings_used >= 3:
                    break

            if len(fib_prices) < 6:
                continue

            fib_prices.sort()
            price = df['close'].iloc[i]
            cluster_threshold = atr_v * 0.5

            for f_idx in range(len(fib_prices) - 2):
                cluster_center = fib_prices[f_idx]
                cluster_count = sum(
                    1 for fp in fib_prices
                    if abs(fp - cluster_center) < cluster_threshold
                )
                if cluster_count >= 3 and abs(price - cluster_center) < atr_v:
                    if price >= cluster_center and df['close'].iloc[i] < df['open'].iloc[i]:
                        sig.iloc[i] = -1
                        break

        return sig


# ═══════════════════════════════════════════════════════════════════════
# 5. SWEEP AND DISPLACE — Stop hunt + immediate violent reversal
# ═══════════════════════════════════════════════════════════════════════

class SweepDisplaceBull(PatternDetector):
    """Stop Hunt below recent equal lows or swing low, immediately followed 
    by a displacement (1-3 bars of strong bullish movement > 1.5x ATR).
    
    Different from LiqSweep_CHoCH which requires a full CHoCH structure.
    This pattern is simpler and faster — just sweep + violent bounce.
    The violence of the displacement is the key signal.
    """
    name = "SweepDisplaceBull"
    category = "confluence"
    LOOKBACK = 30

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK + 10:
            return sig

        atr_vals = utils.atr(df, 14)
        swing_lo_vals = utils.swing_low_values(df, 5)

        for i in range(self.LOOKBACK, len(df)):
            atr_v = atr_vals.iloc[i]
            if pd.isna(atr_v) or atr_v <= 0:
                continue

            # Look for sweep in the last 3 bars
            for offset in range(1, 4):
                j = i - offset
                if j < self.LOOKBACK:
                    continue

                sweep_bar_low = df['low'].iloc[j]
                ref_low = swing_lo_vals.iloc[j]
                if pd.isna(ref_low):
                    continue

                # Sweep: wick below the reference low, but close back above
                swept = (sweep_bar_low < ref_low and
                         df['close'].iloc[j] > ref_low)
                if not swept:
                    continue

                # Displacement: from sweep bar to current bar, measure bullish move
                move_up = df['close'].iloc[i] - df['low'].iloc[j]
                if move_up > atr_v * 1.5:
                    # Verify the displacement bars are bullish
                    bullish_bars = sum(
                        1 for k in range(j, i + 1)
                        if df['close'].iloc[k] > df['open'].iloc[k]
                    )
                    if bullish_bars >= (i - j) * 0.6:  # At least 60% bullish
                        sig.iloc[i] = 1
                        break

        return sig


class SweepDisplaceBear(PatternDetector):
    """Stop Hunt above recent swing high, immediately followed by bearish
    displacement (1-3 bars of strong sell-off > 1.5x ATR).
    """
    name = "SweepDisplaceBear"
    category = "confluence"
    LOOKBACK = 30

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK + 10:
            return sig

        atr_vals = utils.atr(df, 14)
        swing_hi_vals = utils.swing_high_values(df, 5)

        for i in range(self.LOOKBACK, len(df)):
            atr_v = atr_vals.iloc[i]
            if pd.isna(atr_v) or atr_v <= 0:
                continue

            for offset in range(1, 4):
                j = i - offset
                if j < self.LOOKBACK:
                    continue

                sweep_bar_high = df['high'].iloc[j]
                ref_high = swing_hi_vals.iloc[j]
                if pd.isna(ref_high):
                    continue

                swept = (sweep_bar_high > ref_high and
                         df['close'].iloc[j] < ref_high)
                if not swept:
                    continue

                move_down = df['high'].iloc[j] - df['close'].iloc[i]
                if move_down > atr_v * 1.5:
                    bearish_bars = sum(
                        1 for k in range(j, i + 1)
                        if df['close'].iloc[k] < df['open'].iloc[k]
                    )
                    if bearish_bars >= (i - j) * 0.6:
                        sig.iloc[i] = -1
                        break

        return sig


# ═══════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════

ALL_CONFLUENCE = [
    DivAtSupportBull, DivAtResistanceBear,
    DeltaDivBull, DeltaDivBear,
    WyckoffAccumSeq, WyckoffDistribSeq,
    FibClusterBull, FibClusterBear,
    SweepDisplaceBull, SweepDisplaceBear,
]
