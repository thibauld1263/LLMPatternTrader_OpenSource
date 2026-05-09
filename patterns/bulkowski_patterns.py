"""
PatternTrader - Bulkowski Chart Patterns (no look-ahead)
8 patterns derived from Bulkowski's pattern research:

Triangles: Ascending, Descending, Symmetrical, Broadening
Rounding: Rounding Bottom, Rounding Top
Fakey: Bullish Fakey, Bearish Fakey (inside bar → false breakout → reversal)
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils

class AscendingTriangle(PatternDetector):
    """Ascending Triangle: swing highs cluster at flat resistance while
    swing lows make higher lows. Price bounces between a horizontal
    ceiling and a rising support trendline. Breakout above = buy.

    Rules (Bulkowski):
    - Flat top: last N swing highs within ATR tolerance
    - Rising bottom: later swing lows are higher than earlier ones
    - Breakout: close above the resistance level
    """
    name = "AscendingTriangle"
    category = "bulkowski"

    LOOKBACK = 40     # bars to scan for the triangle
    MIN_TOUCHES = 3   # min swing touches on each trendline

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK + 10:
            return sig

        close = df["close"]
        high = df["high"]
        low = df["low"]
        atr_val = utils.atr(df)
        sh_mask = utils.swing_highs(df, 5)
        sl_mask = utils.swing_lows(df, 5)

        for i in range(self.LOOKBACK + 10, len(df)):
            window = slice(i - self.LOOKBACK, i)
            atr_i = atr_val.iloc[i]
            if pd.isna(atr_i) or atr_i <= 0:
                continue

            # Collect swing highs and lows in the window
            sh_idx = [j for j in range(i - self.LOOKBACK, i) if sh_mask.iloc[j]]
            sl_idx = [j for j in range(i - self.LOOKBACK, i) if sl_mask.iloc[j]]

            if len(sh_idx) < self.MIN_TOUCHES or len(sl_idx) < self.MIN_TOUCHES:
                continue

            # Flat top: all swing highs within ATR tolerance
            sh_prices = [high.iloc[j] for j in sh_idx]
            sh_mean = np.mean(sh_prices)
            if max(abs(p - sh_mean) for p in sh_prices) > atr_i * 0.75:
                continue

            # Rising bottom: last swing low > first swing low
            sl_prices = [low.iloc[j] for j in sl_idx]
            if sl_prices[-1] <= sl_prices[0]:
                continue

            # Breakout: close above resistance
            if close.iloc[i] > sh_mean and close.iloc[i - 1] <= sh_mean:
                sig.iloc[i] = 1

        return sig

class DescendingTriangle(PatternDetector):
    """Descending Triangle: swing lows at flat support, swing highs
    make lower highs. Breakdown below support = sell.

    Rules (Bulkowski):
    - Flat bottom: last N swing lows within ATR tolerance
    - Falling top: later swing highs lower than earlier ones
    - Breakdown: close below the support level
    """
    name = "DescendingTriangle"
    category = "bulkowski"

    LOOKBACK = 40
    MIN_TOUCHES = 3

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK + 10:
            return sig

        close = df["close"]
        high = df["high"]
        low = df["low"]
        atr_val = utils.atr(df)
        sh_mask = utils.swing_highs(df, 5)
        sl_mask = utils.swing_lows(df, 5)

        for i in range(self.LOOKBACK + 10, len(df)):
            atr_i = atr_val.iloc[i]
            if pd.isna(atr_i) or atr_i <= 0:
                continue

            sh_idx = [j for j in range(i - self.LOOKBACK, i) if sh_mask.iloc[j]]
            sl_idx = [j for j in range(i - self.LOOKBACK, i) if sl_mask.iloc[j]]

            if len(sh_idx) < self.MIN_TOUCHES or len(sl_idx) < self.MIN_TOUCHES:
                continue

            # Flat bottom
            sl_prices = [low.iloc[j] for j in sl_idx]
            sl_mean = np.mean(sl_prices)
            if max(abs(p - sl_mean) for p in sl_prices) > atr_i * 0.75:
                continue

            # Falling top
            sh_prices = [high.iloc[j] for j in sh_idx]
            if sh_prices[-1] >= sh_prices[0]:
                continue

            # Breakdown
            if close.iloc[i] < sl_mean and close.iloc[i - 1] >= sl_mean:
                sig.iloc[i] = -1

        return sig

class SymmetricalTriangle(PatternDetector):
    """Symmetrical Triangle: both trendlines converge — swing highs
    descend, swing lows ascend. Breakout in either direction.

    Rules (Bulkowski):
    - Converging: swing high slope negative, swing low slope positive
    - 3-3-3-3-3 internal structure (approximated by counting swings)
    - Signal on breakout direction
    """
    name = "SymmetricalTriangle"
    category = "bulkowski"

    LOOKBACK = 40
    MIN_TOUCHES = 3

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK + 10:
            return sig

        close = df["close"]
        high = df["high"]
        low = df["low"]
        atr_val = utils.atr(df)
        sh_mask = utils.swing_highs(df, 5)
        sl_mask = utils.swing_lows(df, 5)

        for i in range(self.LOOKBACK + 10, len(df)):
            atr_i = atr_val.iloc[i]
            if pd.isna(atr_i) or atr_i <= 0:
                continue

            sh_idx = [j for j in range(i - self.LOOKBACK, i) if sh_mask.iloc[j]]
            sl_idx = [j for j in range(i - self.LOOKBACK, i) if sl_mask.iloc[j]]

            if len(sh_idx) < self.MIN_TOUCHES or len(sl_idx) < self.MIN_TOUCHES:
                continue

            sh_prices = [high.iloc[j] for j in sh_idx]
            sl_prices = [low.iloc[j] for j in sl_idx]

            # Converging: highs descending AND lows ascending
            if sh_prices[-1] >= sh_prices[0] or sl_prices[-1] <= sl_prices[0]:
                continue

            # Compute resistance/support at the current bar via linear interpolation
            resistance = sh_prices[-1]
            support = sl_prices[-1]

            # Bullish breakout
            if close.iloc[i] > resistance and close.iloc[i - 1] <= resistance:
                sig.iloc[i] = 1
            # Bearish breakdown
            elif close.iloc[i] < support and close.iloc[i - 1] >= support:
                sig.iloc[i] = -1

        return sig

class BroadeningTriangle(PatternDetector):
    """Broadening Triangle (Reverse Symmetrical): trendlines diverge —
    swing highs ascend, swing lows descend. Reversal pattern.

    Rules (Bulkowski):
    - Diverging: swing high slope positive, swing low slope negative
    - Signal on reversal at extremes (price returns inside)
    """
    name = "BroadeningTriangle"
    category = "bulkowski"

    LOOKBACK = 40
    MIN_TOUCHES = 3

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK + 10:
            return sig

        close = df["close"]
        high = df["high"]
        low = df["low"]
        atr_val = utils.atr(df)
        sh_mask = utils.swing_highs(df, 5)
        sl_mask = utils.swing_lows(df, 5)

        for i in range(self.LOOKBACK + 10, len(df)):
            atr_i = atr_val.iloc[i]
            if pd.isna(atr_i) or atr_i <= 0:
                continue

            sh_idx = [j for j in range(i - self.LOOKBACK, i) if sh_mask.iloc[j]]
            sl_idx = [j for j in range(i - self.LOOKBACK, i) if sl_mask.iloc[j]]

            if len(sh_idx) < self.MIN_TOUCHES or len(sl_idx) < self.MIN_TOUCHES:
                continue

            sh_prices = [high.iloc[j] for j in sh_idx]
            sl_prices = [low.iloc[j] for j in sl_idx]

            # Diverging: highs ascending AND lows descending
            if sh_prices[-1] <= sh_prices[0] or sl_prices[-1] >= sl_prices[0]:
                continue

            upper = sh_prices[-1]
            lower = sl_prices[-1]

            # Bearish reversal at upper extreme
            if (high.iloc[i] >= upper and close.iloc[i] < upper and
                    utils.is_bearish(df).iloc[i]):
                sig.iloc[i] = -1
            # Bullish reversal at lower extreme
            elif (low.iloc[i] <= lower and close.iloc[i] > lower and
                    utils.is_bullish(df).iloc[i]):
                sig.iloc[i] = 1

        return sig

class RoundingBottom(PatternDetector):
    """Rounding Bottom: price forms a U-shaped bowl over ~30+ bars.
    Close above the left rim signals a bullish breakout.

    Rules (Bulkowski):
    - Shape: rounded bowl, lows form a curve
    - Confirmation: close above the left peak (rim)
    - Performance rank: 7 out of 39 (strong)
    """
    name = "RoundingBottom"
    category = "bulkowski"

    LOOKBACK = 40

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK + 10:
            return sig

        close = df["close"]
        low = df["low"]
        atr_val = utils.atr(df)

        for i in range(self.LOOKBACK + 10, len(df)):
            atr_i = atr_val.iloc[i]
            if pd.isna(atr_i) or atr_i <= 0:
                continue

            window_low = low.iloc[i - self.LOOKBACK:i].values
            n = len(window_low)

            # Find bowl shape: minimum in the middle 60%
            min_idx = np.argmin(window_low)
            if min_idx < n * 0.2 or min_idx > n * 0.8:
                continue  # minimum not in the middle = not a bowl

            # Left rim (start) and right rim (end) should be higher than center
            left_rim = np.mean(window_low[:5])
            right_rim = np.mean(window_low[-5:])
            center = window_low[min_idx]

            depth = min(left_rim, right_rim) - center
            if depth < atr_i * 1.5:
                continue  # not deep enough

            # Both rims should be somewhat similar height
            if abs(left_rim - right_rim) > atr_i * 2:
                continue

            # Breakout: close above the left rim
            if close.iloc[i] > left_rim and close.iloc[i - 1] <= left_rim:
                sig.iloc[i] = 1

        return sig

class RoundingTop(PatternDetector):
    """Rounding Top: price forms an inverted U-shape over ~30+ bars.
    Close below the lower rim signals a bearish breakdown.

    Rules (Bulkowski):
    - Shape: rounded dome, highs form an inverted curve
    - Breakdown: close below the lower of the two rims
    - Performance rank: 2-3 out of 39 (very strong)
    """
    name = "RoundingTop"
    category = "bulkowski"

    LOOKBACK = 40

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < self.LOOKBACK + 10:
            return sig

        close = df["close"]
        high = df["high"]
        atr_val = utils.atr(df)

        for i in range(self.LOOKBACK + 10, len(df)):
            atr_i = atr_val.iloc[i]
            if pd.isna(atr_i) or atr_i <= 0:
                continue

            window_high = high.iloc[i - self.LOOKBACK:i].values
            n = len(window_high)

            # Find dome shape: maximum in the middle 60%
            max_idx = np.argmax(window_high)
            if max_idx < n * 0.2 or max_idx > n * 0.8:
                continue

            left_rim = np.mean(window_high[:5])
            right_rim = np.mean(window_high[-5:])
            peak = window_high[max_idx]

            height = peak - max(left_rim, right_rim)
            if height < atr_i * 1.5:
                continue

            if abs(left_rim - right_rim) > atr_i * 2:
                continue

            # Breakdown: close below the lower rim
            lower_rim = min(left_rim, right_rim)
            if close.iloc[i] < lower_rim and close.iloc[i - 1] >= lower_rim:
                sig.iloc[i] = -1

        return sig

class BullishFakey(PatternDetector):
    """Bullish Fakey: 4-candle pattern.
    C1 = mother bar (defines range)
    C2 = inside bar (C2 high < C1 high AND C2 low > C1 low)
    C3 = false breakout DOWN (C3 low < C1 low — stop hunt)
    C4 = reversal UP (C4 close > C1 high — reclaim)

    The false breakout traps bears, then reverses strongly.
    Bulkowski: 47% break-even failure rate, 22% avg rise.
    """
    name = "BullishFakey"
    category = "bulkowski"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < 10:
            return sig

        o = df["open"].values
        c = df["close"].values
        h = df["high"].values
        l = df["low"].values

        for i in range(3, len(df)):
            # C1 = i-3, C2 = i-2, C3 = i-1, C4 = i
            c1_h, c1_l = h[i - 3], l[i - 3]
            c2_h, c2_l = h[i - 2], l[i - 2]
            c3_l = l[i - 1]
            c4_c = c[i]

            # Inside bar: C2 fits inside C1 (strict, no ties)
            if c2_h >= c1_h or c2_l <= c1_l:
                continue

            # False breakout down: C3 breaks below C1 low
            if c3_l >= c1_l:
                continue

            # Reversal: C4 closes above C1 high
            if c4_c > c1_h:
                sig.iloc[i] = 1

        return sig

class BearishFakey(PatternDetector):
    """Bearish Fakey: 4-candle pattern.
    C1 = mother bar
    C2 = inside bar
    C3 = false breakout UP (C3 high > C1 high)
    C4 = reversal DOWN (C4 close < C1 low)

    The false breakout traps bulls, then reverses strongly.
    Bulkowski: 56% break-even failure rate, 8% avg decline.
    """
    name = "BearishFakey"
    category = "bulkowski"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < 10:
            return sig

        o = df["open"].values
        c = df["close"].values
        h = df["high"].values
        l = df["low"].values

        for i in range(3, len(df)):
            c1_h, c1_l = h[i - 3], l[i - 3]
            c2_h, c2_l = h[i - 2], l[i - 2]
            c3_h = h[i - 1]
            c4_c = c[i]

            # Inside bar
            if c2_h >= c1_h or c2_l <= c1_l:
                continue

            # False breakout up
            if c3_h <= c1_h:
                continue

            # Reversal: C4 closes below C1 low
            if c4_c < c1_l:
                sig.iloc[i] = -1

        return sig

ALL_BULKOWSKI = [
    AscendingTriangle, DescendingTriangle,
    SymmetricalTriangle, BroadeningTriangle,
    RoundingBottom, RoundingTop,
    BullishFakey, BearishFakey,
]
