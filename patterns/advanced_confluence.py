"""
Advanced Confluence Patterns (Super Patterns)
These patterns combine Market Structure, Liquidity, Momentum, and Session Timing.

Patterns included:
1. London Killzone Sweep
2. Holy Grail Pullback (Single TF)
3. Multi Divergence (RSI + MACD)
4. Hidden Divergence
5. US Open Range Breakout
6. Inside Bar Breakout
7. ADX Trend Ignition
8. Mean Reversion BB
9. Momentum Shift
10. Narrow Range Expansion
11. Volume Price Divergence
12. OBFVG Confluence
13. EMARibbon Trend
14. Power of Three
15. VWAP Bounce
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils

class LondonKillzoneSweep(PatternDetector):
    name = "LondonKillzoneSweep"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 50:
            return signals

        # We need UTC hours
        hours = df.index.hour
        
        # Track Asian High/Low for each day
        asian_highs = pd.Series(np.nan, index=df.index)
        asian_lows = pd.Series(np.nan, index=df.index)
        
        current_day = None
        curr_high = -np.inf
        curr_low = np.inf
        
        # Optimized rolling tracking for Asian Range
        for i, (idx, row) in enumerate(df.iterrows()):
            h = hours[i]
            d = idx.date()
            if d != current_day:
                current_day = d
                curr_high = -np.inf
                curr_low = np.inf
                
            if 0 <= h < 7:
                curr_high = max(curr_high, row['high'])
                curr_low = min(curr_low, row['low'])
            
            # Store the range values for the rest of the day
            if h >= 7 and curr_high != -np.inf:
                asian_highs.iloc[i] = curr_high
                asian_lows.iloc[i] = curr_low
                
        # Now detect the sweep during London Killzone (07:00 - 10:00)
        # We need:
        # 1. High breaks Asian High, then closes below it (Bearish SFP) -> SELL
        # 2. Low breaks Asian Low, then closes above it (Bullish SFP) -> BUY
        
        in_killzone = (hours >= 7) & (hours <= 10)
        
        # Bearish Sweep: high > asian_high, but close < asian_high, and close < open (red candle)
        bearish_sweep = in_killzone & (df['high'] > asian_highs) & (df['close'] < asian_highs) & (df['close'] < df['open'])
        
        # Bullish Sweep: low < asian_low, but close > asian_low, and close > open (green candle)
        bullish_sweep = in_killzone & (df['low'] < asian_lows) & (df['close'] > asian_lows) & (df['close'] > df['open'])

        signals.loc[bullish_sweep] = 1
        signals.loc[bearish_sweep] = -1
        
        return signals

class HolyGrailPullback(PatternDetector):
    name = "HolyGrailPullback"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 200:
            return signals

        ema50 = utils.ema(df['close'], 50)
        ema200 = utils.ema(df['close'], 200)
        macd_line, macd_signal, macd_hist = utils.macd(df)
        
        # Trend
        uptrend = ema50 > ema200
        downtrend = ema50 < ema200
        
        # Pullback (3 consecutive candles against trend)
        is_red = df['close'] < df['open']
        is_green = df['close'] > df['open']
        three_reds = is_red & is_red.shift(1) & is_red.shift(2)
        three_greens = is_green & is_green.shift(1) & is_green.shift(2)
        
        # Touch EMAs
        touch_ema_bull = (df['low'] <= ema50) & (df['close'] >= ema200)
        touch_ema_bear = (df['high'] >= ema50) & (df['close'] <= ema200)
        
        # MACD slope
        macd_slope_up = (macd_hist < 0) & (macd_hist > macd_hist.shift(1))
        macd_slope_down = (macd_hist > 0) & (macd_hist < macd_hist.shift(1))
        
        # Rejection candle (Pinbar)
        body = abs(df['close'] - df['open'])
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        
        bull_pin = (lower_wick > body * 2) & (upper_wick < body)
        bear_pin = (upper_wick > body * 2) & (lower_wick < body)
        
        bullish_grail = uptrend & (three_reds.shift(1) | three_reds.shift(2)) & touch_ema_bull & macd_slope_up & bull_pin
        bearish_grail = downtrend & (three_greens.shift(1) | three_greens.shift(2)) & touch_ema_bear & macd_slope_down & bear_pin
        
        signals.loc[bullish_grail] = 1
        signals.loc[bearish_grail] = -1
        
        return signals

class MultiDivergence(PatternDetector):
    name = "MultiDivergence"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 30:
            return signals

        rsi = utils.rsi(df, 14)
        _, _, macd_hist = utils.macd(df)
        
        # Rolling min/max to find peaks/troughs
        lows = df['low'].rolling(10).min()
        highs = df['high'].rolling(10).max()
        
        # Bullish: Price LL, RSI HL, MACD HL
        price_ll = df['low'] < lows.shift(10)
        rsi_hl = rsi > rsi.shift(10).rolling(5).min()
        macd_hl = macd_hist > macd_hist.shift(10).rolling(5).min()
        rsi_oversold = rsi < 35
        
        bearish_candle = df['close'] < df['open']
        bullish_candle = df['close'] > df['open']
        
        bullish_div = price_ll & rsi_hl & macd_hl & rsi_oversold & bullish_candle
        
        # Bearish: Price HH, RSI LL, MACD LL
        price_hh = df['high'] > highs.shift(10)
        rsi_ll = rsi < rsi.shift(10).rolling(5).max()
        macd_ll = macd_hist < macd_hist.shift(10).rolling(5).max()
        rsi_overbought = rsi > 65
        
        bearish_div = price_hh & rsi_ll & macd_ll & rsi_overbought & bearish_candle
        
        signals.loc[bullish_div] = 1
        signals.loc[bearish_div] = -1
        
        return signals

class HiddenDivergence(PatternDetector):
    name = "HiddenDivergence"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 30:
            return signals

        rsi = utils.rsi(df, 14)
        ema50 = utils.ema(df['close'], 50)
        
        # Define rolling windows to capture recent structure
        past_low = df['low'].shift(10).rolling(10).min()
        past_high = df['high'].shift(10).rolling(10).max()
        past_rsi_low = rsi.shift(10).rolling(10).min()
        past_rsi_high = rsi.shift(10).rolling(10).max()
        
        # Bullish Hidden: Uptrend (Close > EMA50), Price HL, RSI LL
        uptrend = df['close'] > ema50
        price_hl = df['low'] > past_low
        rsi_ll = rsi < past_rsi_low
        bullish_candle = df['close'] > df['open']
        
        bullish_hidden = uptrend & price_hl & rsi_ll & bullish_candle
        
        # Bearish Hidden: Downtrend (Close < EMA50), Price LH, RSI HH
        downtrend = df['close'] < ema50
        price_lh = df['high'] < past_high
        rsi_hh = rsi > past_rsi_high
        bearish_candle = df['close'] < df['open']
        
        bearish_hidden = downtrend & price_lh & rsi_hh & bearish_candle
        
        signals.loc[bullish_hidden] = 1
        signals.loc[bearish_hidden] = -1
        
        return signals

class USOpenRangeBreakout(PatternDetector):
    name = "USOpenRangeBreakout"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 50:
            return signals

        # Estimate Timeframe in hours
        delta = df.index[-1] - df.index[-2]
        tf_hours = delta.total_seconds() / 3600.0

        current_day = None
        us_high = -np.inf
        us_low = np.inf
        breakout_done = False
        
        for i, (idx, row) in enumerate(df.iterrows()):
            h = idx.hour
            d = idx.date()
            if d != current_day:
                current_day = d
                us_high = -np.inf
                us_low = np.inf
                breakout_done = False
                
            # If TF is H4 or higher, the 12:00 UTC candle builds the range
            if tf_hours >= 4:
                can_build = (h == 12)
                can_breakout = (h >= 16)
            else:
                # For H1, M30, M15, M5: use 13:00 to 14:59 UTC
                can_build = (13 <= h <= 14)
                can_breakout = (h >= 15)
                
            if can_build:
                us_high = max(us_high, row['high'])
                us_low = min(us_low, row['low'])
            
            if can_breakout and not breakout_done and us_high != -np.inf:
                # Strong breakout confirmation: body > 50% of the candle range
                body = abs(row['close'] - row['open'])
                rng = row['high'] - row['low']
                strong_momentum = body > (rng * 0.5)

                # Check for Breakout above US High
                if row['close'] > us_high and row['close'] > row['open']:
                    if strong_momentum:
                        signals.iloc[i] = 1
                        breakout_done = True
                # Check for Breakout below US Low
                elif row['close'] < us_low and row['close'] < row['open']:
                    if strong_momentum:
                        signals.iloc[i] = -1
                        breakout_done = True

        return signals

class InsideBarBreakout(PatternDetector):
    """Inside Bar (mother candle engulfs child) followed by a breakout candle
    with ADX confirmation (trending environment > 20)."""
    name = "InsideBarBreakout"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 30:
            return signals
        adx, pdi, mdi = utils.adx(df, 14)
        # Inside Bar: current high < prev high AND current low > prev low
        inside = (df['high'].shift(1) < df['high'].shift(2)) & (df['low'].shift(1) > df['low'].shift(2))
        # Breakout candle (current)
        bull_break = inside & (df['close'] > df['high'].shift(2)) & (df['close'] > df['open']) & (adx > 20)
        bear_break = inside & (df['close'] < df['low'].shift(2)) & (df['close'] < df['open']) & (adx > 20)
        signals.loc[bull_break] = 1
        signals.loc[bear_break] = -1
        return signals

class ADXTrendIgnition(PatternDetector):
    """ADX crosses above 25 from below while +DI/-DI shows clear direction.
    Catches the very start of a new strong trend."""
    name = "ADXTrendIgnition"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 30:
            return signals
        adx, pdi, mdi = utils.adx(df, 14)
        adx_cross_up = (adx > 25) & (adx.shift(1) <= 25)
        bull_ignition = adx_cross_up & (pdi > mdi) & (df['close'] > df['open'])
        bear_ignition = adx_cross_up & (mdi > pdi) & (df['close'] < df['open'])
        signals.loc[bull_ignition] = 1
        signals.loc[bear_ignition] = -1
        return signals

class MeanReversionBB(PatternDetector):
    """Price touches outer Bollinger Band in a RANGING market (ADX < 20),
    then the next candle closes back inside. Classic mean reversion."""
    name = "MeanReversionBB"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 30:
            return signals
        up, mid, low = utils.bollinger(df, 20, 2.0)
        adx, _, _ = utils.adx(df, 14)
        ranging = adx < 20
        # Prev candle touched/broke lower BB, current closes back inside
        bull_revert = ranging & (df['low'].shift(1) <= low.shift(1)) & (df['close'] > low) & (df['close'] > df['open'])
        # Prev candle touched/broke upper BB, current closes back inside
        bear_revert = ranging & (df['high'].shift(1) >= up.shift(1)) & (df['close'] < up) & (df['close'] < df['open'])
        signals.loc[bull_revert] = 1
        signals.loc[bear_revert] = -1
        return signals

class MomentumShift(PatternDetector):
    """MACD crosses signal line while RSI confirms direction from neutral zone.
    Catches momentum shifts early."""
    name = "MomentumShift"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 30:
            return signals
        rsi = utils.rsi(df, 14)
        macd_line, macd_signal, macd_hist = utils.macd(df)
        # MACD cross up + RSI rising from below 50
        bull_cross = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
        bull_rsi = (rsi > 45) & (rsi < 65)  # Not overbought
        bear_cross = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))
        bear_rsi = (rsi < 55) & (rsi > 35)  # Not oversold
        signals.loc[bull_cross & bull_rsi] = 1
        signals.loc[bear_cross & bear_rsi] = -1
        return signals

class NarrowRangeExpansion(PatternDetector):
    """NR7: Narrowest range of the last 7 bars followed by an expansion candle.
    Volatility contraction leads to expansion."""
    name = "NarrowRangeExpansion"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 10:
            return signals
        rng = df['high'] - df['low']
        min_7 = rng.rolling(7).min()
        # Previous bar was the narrowest of last 7
        was_nr7 = rng.shift(1) == min_7.shift(1)
        # Current bar is an expansion (range > 1.5x the NR bar)
        expansion = rng > (rng.shift(1) * 1.5)
        bull_exp = was_nr7 & expansion & (df['close'] > df['open'])
        bear_exp = was_nr7 & expansion & (df['close'] < df['open'])
        signals.loc[bull_exp] = 1
        signals.loc[bear_exp] = -1
        return signals

class VolumePriceDivergence(PatternDetector):
    """Price makes new high/low but volume decreases — sign of exhaustion.
    Smart money is exiting while retail pushes further."""
    name = "VolumePriceDivergence"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 20:
            return signals
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
        if vol_col not in df.columns:
            return signals
        vol = df[vol_col]
        # Price HH but volume lower than previous HH's volume
        price_hh = df['high'] > df['high'].shift(1).rolling(10).max()
        vol_lower = vol < vol.shift(1).rolling(10).max() * 0.7
        bear_exhaust = price_hh & vol_lower & (df['close'] < df['open'])
        # Price LL but volume lower
        price_ll = df['low'] < df['low'].shift(1).rolling(10).min()
        vol_lower_bull = vol < vol.shift(1).rolling(10).max() * 0.7
        bull_exhaust = price_ll & vol_lower_bull & (df['close'] > df['open'])
        signals.loc[bull_exhaust] = 1
        signals.loc[bear_exhaust] = -1
        return signals

class OBFVGConfluence(PatternDetector):
    """Order Block that sits inside a Fair Value Gap — extreme confluence zone.
    When both OB and FVG overlap, institutions are very likely to defend the level."""
    name = "OBFVGConfluence"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 10:
            return signals
        # Bullish OB: last red candle before a strong green move
        strong_green = (df['close'] - df['open']) > (df['high'] - df['low']).rolling(10).mean() * 1.5
        prev_red = df['close'].shift(1) < df['open'].shift(1)
        ob_bull_zone_high = df['open'].shift(1)
        ob_bull_zone_low = df['close'].shift(1)
        # FVG Bull: gap between bar[-2] high and bar[0] low
        fvg_bull = df['low'] > df['high'].shift(2)
        # Confluence: OB zone overlaps with FVG zone on the current retest
        bull_conf = strong_green.shift(-1) if len(df) > 2 else pd.Series(False, index=df.index)
        # Simplified: strong impulse after OB + FVG present
        prev2_high = df['high'].shift(2)
        curr_low = df['low']
        has_fvg_bull = curr_low > prev2_high
        bull_signal = prev_red & strong_green & has_fvg_bull
        # Bearish
        strong_red = (df['open'] - df['close']) > (df['high'] - df['low']).rolling(10).mean() * 1.5
        prev_green = df['close'].shift(1) > df['open'].shift(1)
        has_fvg_bear = df['high'] < df['low'].shift(2)
        bear_signal = prev_green & strong_red & has_fvg_bear
        signals.loc[bull_signal] = 1
        signals.loc[bear_signal] = -1
        return signals

class EMARibbonTrend(PatternDetector):
    """EMA Ribbon (8, 13, 21, 34, 55) perfectly stacked + price pullback to the ribbon.
    When all EMAs are aligned and price dips into the ribbon, very high probability continuation."""
    name = "EMARibbonTrend"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 60:
            return signals
        e8 = utils.ema(df['close'], 8)
        e13 = utils.ema(df['close'], 13)
        e21 = utils.ema(df['close'], 21)
        e34 = utils.ema(df['close'], 34)
        e55 = utils.ema(df['close'], 55)
        # Bull ribbon: 8 > 13 > 21 > 34 > 55
        bull_stack = (e8 > e13) & (e13 > e21) & (e21 > e34) & (e34 > e55)
        # Price pulls back INTO the ribbon (touches e21 or e34)
        bull_pullback = (df['low'] <= e21) & (df['close'] > e13) & (df['close'] > df['open'])
        # Bear ribbon: 8 < 13 < 21 < 34 < 55
        bear_stack = (e8 < e13) & (e13 < e21) & (e21 < e34) & (e34 < e55)
        bear_pullback = (df['high'] >= e21) & (df['close'] < e13) & (df['close'] < df['open'])
        signals.loc[bull_stack & bull_pullback] = 1
        signals.loc[bear_stack & bear_pullback] = -1
        return signals

class PowerOfThree(PatternDetector):
    """ICT Power of 3 (AMD): Accumulation → Manipulation → Distribution.
    Detects the manipulation phase (false sweep) followed by the distribution (real move).
    Works on any TF by looking at the first third of each day's candles."""
    name = "PowerOfThree"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 50:
            return signals
        hours = df.index.hour
        # Phase 1: Accumulation (00:00 - 06:00) — build a range
        # Phase 2: Manipulation (06:00 - 09:00) — false break of the range
        # Phase 3: Distribution (09:00+) — the real move
        current_day = None
        acc_high = -np.inf
        acc_low = np.inf
        manip_done_bull = False
        manip_done_bear = False

        for i, (idx, row) in enumerate(df.iterrows()):
            h = hours[i]
            d = idx.date()
            if d != current_day:
                current_day = d
                acc_high = -np.inf
                acc_low = np.inf
                manip_done_bull = False
                manip_done_bear = False

            if 0 <= h < 6:
                acc_high = max(acc_high, row['high'])
                acc_low = min(acc_low, row['low'])
            elif 6 <= h < 9 and acc_high != -np.inf:
                # Manipulation: sweep above then reject (bearish manip)
                if row['high'] > acc_high and row['close'] < acc_high and row['close'] < row['open']:
                    manip_done_bear = True
                # Manipulation: sweep below then reject (bullish manip)
                if row['low'] < acc_low and row['close'] > acc_low and row['close'] > row['open']:
                    manip_done_bull = True
            elif h >= 9 and acc_high != -np.inf:
                # Distribution: strong move in the opposite direction of manipulation
                body = abs(row['close'] - row['open'])
                rng = row['high'] - row['low']
                if rng > 0 and body > rng * 0.6:
                    if manip_done_bull and row['close'] > row['open'] and not manip_done_bear:
                        signals.iloc[i] = 1
                        manip_done_bull = False  # One signal per day
                    elif manip_done_bear and row['close'] < row['open'] and not manip_done_bull:
                        signals.iloc[i] = -1
                        manip_done_bear = False
        return signals

class VWAPBounce(PatternDetector):
    """Volume-Weighted Average Price approximation bounce.
    Uses cumulative (Volume * TypicalPrice) / cumVolume per day.
    Signal triggers when price rejects off VWAP in a trending market."""
    name = "VWAPBounce"
    category = "AdvancedConfluence"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(df) < 30:
            return signals
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
        if vol_col not in df.columns:
            return signals

        ema50 = utils.ema(df['close'], 50)
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        vol = df[vol_col].astype(float)

        # Calculate intraday VWAP
        vwap = pd.Series(np.nan, index=df.index)
        cum_tpv = 0.0
        cum_vol = 0.0
        current_day = None
        for i, (idx, row) in enumerate(df.iterrows()):
            d = idx.date()
            if d != current_day:
                current_day = d
                cum_tpv = 0.0
                cum_vol = 0.0
            v = vol.iloc[i]
            cum_tpv += tp.iloc[i] * v
            cum_vol += v
            vwap.iloc[i] = cum_tpv / cum_vol if cum_vol > 0 else tp.iloc[i]

        uptrend = df['close'] > ema50
        downtrend = df['close'] < ema50
        body = abs(df['close'] - df['open'])
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        atr_prox = utils.atr(df, 14) * 0.3  # ATR-based proximity instead of % of price

        # Bull: uptrend, price dips to VWAP, rejection (long lower wick)
        near_vwap_bull = (df['low'] <= vwap + atr_prox) & (df['close'] > vwap)
        bull_rej = lower_wick > body * 1.5
        bull_signal = uptrend & near_vwap_bull & bull_rej

        # Bear: downtrend, price rallies to VWAP, rejection (long upper wick)
        near_vwap_bear = (df['high'] >= vwap - atr_prox) & (df['close'] < vwap)
        bear_rej = upper_wick > body * 1.5
        bear_signal = downtrend & near_vwap_bear & bear_rej

        signals.loc[bull_signal] = 1
        signals.loc[bear_signal] = -1
        return signals

ALL_ADVANCED_CONFLUENCE = [
    LondonKillzoneSweep,
    HolyGrailPullback,
    MultiDivergence,
    HiddenDivergence,
    USOpenRangeBreakout,
    InsideBarBreakout,
    ADXTrendIgnition,
    MeanReversionBB,
    MomentumShift,
    NarrowRangeExpansion,
    VolumePriceDivergence,
    OBFVGConfluence,
    EMARibbonTrend,
    PowerOfThree,
    VWAPBounce,
]

