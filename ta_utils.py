"""
PatternTrader - Utilities
"""

import numpy as np
import pandas as pd
from typing import Tuple
import config

# ── Moving Averages ──────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

# ── ATR ──────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = None) -> pd.Series:
    if period is None:
        period = config.ATR_PERIOD
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

# ── RSI ──────────────────────────────────────────────────────────────

def rsi(df: pd.DataFrame, period: int = None) -> pd.Series:
    if period is None:
        period = config.RSI_PERIOD
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

# ── MACD ─────────────────────────────────────────────────────────────

def macd(df: pd.DataFrame, fast=None, slow=None, signal=None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast = fast or config.MACD_FAST
    slow = slow or config.MACD_SLOW
    signal = signal or config.MACD_SIGNAL
    close = df["close"]
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# ── Bollinger Bands ──────────────────────────────────────────────────

def bollinger(df: pd.DataFrame, period=None, std_dev=None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    period = period or config.BOLLINGER_PERIOD
    std_dev = std_dev or config.BOLLINGER_STD
    mid = sma(df["close"], period)
    std = df["close"].rolling(period).std()
    return mid + std_dev * std, mid, mid - std_dev * std

# ── ADX ──────────────────────────────────────────────────────────────

def adx(df: pd.DataFrame, period: int = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    period = period or config.ADX_PERIOD
    high, low, close = df["high"], df["low"], df["close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr = atr(df, period)
    alpha = 1 / period
    smooth_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = 100 * smooth_plus / tr.replace(0, np.nan)
    minus_di = 100 * smooth_minus / tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx_val, plus_di, minus_di

# ── Swing Detection (CAUSAL - no look-ahead) ────────────────────────
# A swing at bar i needs `order` bars on each side.
# Flag placed at bar i+order (confirmation bar) to avoid future leak.

def swing_highs(df: pd.DataFrame, order: int = None) -> pd.Series:
    order = order or config.SWING_ORDER
    high = df["high"]
    result = pd.Series(False, index=df.index)
    for i in range(order, len(high) - order):
        window = high.iloc[i - order : i + order + 1]
        if high.iloc[i] == window.max():
            c = i + order
            if c < len(high):
                result.iloc[c] = True
    return result

def swing_lows(df: pd.DataFrame, order: int = None) -> pd.Series:
    order = order or config.SWING_ORDER
    low = df["low"]
    result = pd.Series(False, index=df.index)
    for i in range(order, len(low) - order):
        window = low.iloc[i - order : i + order + 1]
        if low.iloc[i] == window.min():
            c = i + order
            if c < len(low):
                result.iloc[c] = True
    return result

def swing_high_values(df: pd.DataFrame, order: int = None) -> pd.Series:
    order = order or config.SWING_ORDER
    mask = swing_highs(df, order)
    high = df["high"]
    vals = pd.Series(np.nan, index=df.index)
    for c in np.where(mask)[0]:
        sb = c - order
        if 0 <= sb < len(high):
            vals.iloc[c] = high.iloc[sb]
    return vals.ffill()

def swing_low_values(df: pd.DataFrame, order: int = None) -> pd.Series:
    order = order or config.SWING_ORDER
    mask = swing_lows(df, order)
    low = df["low"]
    vals = pd.Series(np.nan, index=df.index)
    for c in np.where(mask)[0]:
        sb = c - order
        if 0 <= sb < len(low):
            vals.iloc[c] = low.iloc[sb]
    return vals.ffill()

# ── Candle helpers ───────────────────────────────────────────────────

def body_size(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()

def upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)

def lower_shadow(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]

def candle_range(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df["low"]

def is_bullish(df: pd.DataFrame) -> pd.Series:
    return df["close"] > df["open"]

def is_bearish(df: pd.DataFrame) -> pd.Series:
    return df["close"] < df["open"]

def avg_body(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return body_size(df).rolling(period).mean()

def avg_range(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return candle_range(df).rolling(period).mean()

# ── Support / Resistance Zones ───────────────────────────────────────

def sr_zones(df: pd.DataFrame, lookback: int = 100, num_zones: int = 3,
             cluster_pct: float = 0.15) -> dict:
    """Detect S/R zones from recent swing highs/lows over `lookback` bars.
    
    Algorithm:
    1. Find local highs and lows using a rolling window (order=5).
    2. Collect all swing prices.
    3. Cluster nearby levels within `cluster_pct`% of ATR.
    4. Separate into resistance (above current price) and support (below).
    5. Return the closest `num_zones` in each direction.
    
    Returns dict with keys: 
        resistance_1, resistance_2, resistance_3 (closest → farthest above price)
        support_1, support_2, support_3 (closest → farthest below price)
    """
    if len(df) < lookback:
        lookback = len(df)
    
    recent = df.iloc[-lookback:]
    current_price = recent['close'].iloc[-1]
    
    # ATR for clustering threshold
    atr_val = atr(recent, 14).iloc[-1] if len(recent) >= 14 else (recent['high'] - recent['low']).mean()
    if pd.isna(atr_val) or atr_val <= 0:
        atr_val = (recent['high'] - recent['low']).mean()
    
    cluster_dist = atr_val * cluster_pct
    
    # Collect swing highs and lows with a small order window
    order = 5
    levels = []
    
    for i in range(order, len(recent) - order):
        # Swing high
        high_window = recent['high'].iloc[i - order: i + order + 1]
        if recent['high'].iloc[i] == high_window.max():
            levels.append(recent['high'].iloc[i])
        
        # Swing low
        low_window = recent['low'].iloc[i - order: i + order + 1]
        if recent['low'].iloc[i] == low_window.min():
            levels.append(recent['low'].iloc[i])
    
    if not levels:
        return {}
    
    # Sort and cluster nearby levels
    levels.sort()
    clusters = []
    current_cluster = [levels[0]]
    
    for lvl in levels[1:]:
        if abs(lvl - current_cluster[-1]) <= cluster_dist:
            current_cluster.append(lvl)
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [lvl]
    clusters.append(np.mean(current_cluster))
    
    # Separate into support (below price) and resistance (above price)
    # Use a small buffer so levels right at the price aren't double-counted
    buffer = atr_val * 0.05
    resistance = sorted([c for c in clusters if c > current_price + buffer])
    support = sorted([c for c in clusters if c < current_price - buffer], reverse=True)
    
    # Determine precision from data
    digits = 5
    if current_price > 100:   # Indices, JPY crosses
        digits = 2
    elif current_price > 10:
        digits = 3
    
    result = {}
    for i in range(num_zones):
        key_r = f"resistance_{i+1}"
        key_s = f"support_{i+1}"
        result[key_r] = round(resistance[i], digits) if i < len(resistance) else None
        result[key_s] = round(support[i], digits) if i < len(support) else None
    
    return result

# ── VWAP ─────────────────────────────────────────────────────────────

def vwap(df: pd.DataFrame) -> float:
    """Daily-reset VWAP (Volume-Weighted Average Price) for the current bar.
    Uses tick_volume (stored as 'volume' column)."""
    typical = (df['high'] + df['low'] + df['close']) / 3
    vol = df['volume'].replace(0, np.nan)
    date_groups = df.index.date
    cum_tp_vol = (typical * vol).groupby(date_groups).cumsum()
    cum_vol = vol.groupby(date_groups).cumsum()
    vwap_series = cum_tp_vol / cum_vol
    val = vwap_series.iloc[-1] if len(vwap_series) > 0 else np.nan
    return round(val, 5) if pd.notna(val) else None
