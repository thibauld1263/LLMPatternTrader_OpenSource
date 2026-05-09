"""
PatternTrader - Multi-Structure Combo Patterns (no look-ahead)
10 signals that combine multiple market structure events:

1-2 Reversal: Two consecutive CHoCH events = confirmed reversal.
CHoCH + OB Retest: Change of Character followed by retest of Order Block.
BOS + FVG Fill: Break of Structure impulse leaves a gap; entry on fill.
Mitigation Block: Failed Order Block flips polarity; entry on retest.
Liquidity Sweep + CHoCH: Stop hunt immediately followed by structure change.
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils

def _choch_events(df: pd.DataFrame, order: int = None):
    """
    Return two boolean Series: (bullish_choch, bearish_choch).

    Bullish CHoCH: price was making lower highs (downtrend), then
    close breaks above the most recent swing high = first break
    against the prevailing trend.

    Bearish CHoCH: price was making higher lows (uptrend), then
    close breaks below the most recent swing low.

    Uses causal swing detection — no look-ahead.
    """
    swing_h_mask = utils.swing_highs(df, order)
    swing_l_mask = utils.swing_lows(df, order)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    o = order or 5

    # Build swing-high and swing-low value series (forward-filled)
    sh_vals = utils.swing_high_values(df, order)
    sl_vals = utils.swing_low_values(df, order)

    # Track whether the last two swing highs are descending (downtrend)
    # and last two swing lows are ascending (uptrend)
    prev_sh = sh_vals.shift(1)
    prev_sl = sl_vals.shift(1)

    # Downtrend context: current swing high < previous swing high
    # Use rolling approach: the last confirmed swing high is lower than
    # the one before it.
    in_downtrend = sh_vals < prev_sh
    in_uptrend = sl_vals > prev_sl

    # Forward-fill trend context so it persists between swings
    in_downtrend = in_downtrend.ffill().fillna(False)
    in_uptrend = in_uptrend.ffill().fillna(False)

    # Bullish CHoCH: in downtrend and close breaks above last swing high
    prev_close = close.shift(1)
    bull_choch = (
        in_downtrend &
        (prev_close <= sh_vals.shift(1)) &
        (close > sh_vals) &
        sh_vals.notna() &
        utils.is_bullish(df)
    )

    # Bearish CHoCH: in uptrend and close breaks below last swing low
    bear_choch = (
        in_uptrend &
        (prev_close >= sl_vals.shift(1)) &
        (close < sl_vals) &
        sl_vals.notna() &
        utils.is_bearish(df)
    )

    return bull_choch, bear_choch

class OneTwoReversalBull(PatternDetector):
    """1-2 Reversal Bullish: a Bearish CHoCH followed by a Bullish CHoCH
    within 30 bars. The first CHoCH signals exhaustion of the up-move;
    the second CHoCH confirms the new bullish direction. Entry is on
    the second (bullish) CHoCH candle.

    Classic 1-2 reversal pattern used in SMC/ICT trading."""
    name = "1-2_ReversalBull"
    category = "multi_structure"

    LOOKBACK = 30  # max bars between the two CHoCH events

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < 50:
            return sig

        bull_choch, bear_choch = _choch_events(df)

        # For each bullish CHoCH, check if a bearish CHoCH happened
        # within the preceding LOOKBACK bars
        bear_recently = bear_choch.rolling(self.LOOKBACK, min_periods=1).max().shift(1)

        # Signal fires when: bullish CHoCH AND there was a bearish CHoCH
        # within the last LOOKBACK bars
        combo = bull_choch & (bear_recently > 0)
        sig[combo] = 1
        return sig

class OneTwoReversalBear(PatternDetector):
    """1-2 Reversal Bearish: Bullish CHoCH followed by Bearish CHoCH
    within 30 bars. Entry on the second (bearish) CHoCH."""
    name = "1-2_ReversalBear"
    category = "multi_structure"

    LOOKBACK = 30

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < 50:
            return sig

        bull_choch, bear_choch = _choch_events(df)

        bull_recently = bull_choch.rolling(self.LOOKBACK, min_periods=1).max().shift(1)

        combo = bear_choch & (bull_recently > 0)
        sig[combo] = -1
        return sig

class CHoCHOBRetestBull(PatternDetector):
    """CHoCH + OB Retest Bullish: a bullish CHoCH occurs, marking the
    last bearish candle before the CHoCH impulse as an Order Block.
    When price returns to that OB zone = high-probability buy entry.

    The CHoCH gives trend-change context; the OB gives precise entry."""
    name = "CHoCH_OBRetestBull"
    category = "multi_structure"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < 50:
            return sig

        bull_choch, _ = _choch_events(df)
        o, c, high, low = df["open"], df["close"], df["high"], df["low"]

        # Track the OB zone created by each bullish CHoCH
        ob_high = pd.Series(np.nan, index=df.index)
        ob_low = pd.Series(np.nan, index=df.index)

        for i in range(2, len(df)):
            if not bull_choch.iloc[i]:
                continue
            # Find the last bearish candle before the CHoCH candle
            for j in range(i - 1, max(i - 5, 0), -1):
                if c.iloc[j] < o.iloc[j]:  # bearish candle
                    ob_high.iloc[i] = o.iloc[j]  # open = top of OB
                    ob_low.iloc[i] = c.iloc[j]   # close = bottom of OB
                    break

        # Forward-fill OB zones so they persist
        ob_high = ob_high.ffill()
        ob_low = ob_low.ffill()

        # Signal when price returns to OB zone
        touches = (low <= ob_high) & (c > ob_low) & ob_high.notna()
        # Avoid re-triggering on the same OB zone
        first_touch = touches & ~touches.shift(1).fillna(False).astype(bool)
        # Don't signal on the CHoCH bar itself
        first_touch = first_touch & ~bull_choch

        sig[first_touch & utils.is_bullish(df)] = 1
        return sig

class CHoCHOBRetestBear(PatternDetector):
    """CHoCH + OB Retest Bearish: bearish CHoCH → retest of the last
    bullish candle (OB zone) before the CHoCH impulse = sell entry."""
    name = "CHoCH_OBRetestBear"
    category = "multi_structure"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < 50:
            return sig

        _, bear_choch = _choch_events(df)
        o, c, high, low = df["open"], df["close"], df["high"], df["low"]

        ob_high = pd.Series(np.nan, index=df.index)
        ob_low = pd.Series(np.nan, index=df.index)

        for i in range(2, len(df)):
            if not bear_choch.iloc[i]:
                continue
            for j in range(i - 1, max(i - 5, 0), -1):
                if c.iloc[j] > o.iloc[j]:  # bullish candle = bearish OB
                    ob_high.iloc[i] = c.iloc[j]
                    ob_low.iloc[i] = o.iloc[j]
                    break

        ob_high = ob_high.ffill()
        ob_low = ob_low.ffill()

        touches = (high >= ob_low) & (c < ob_high) & ob_low.notna()
        first_touch = touches & ~touches.shift(1).fillna(False).astype(bool)
        first_touch = first_touch & ~bear_choch

        sig[first_touch & utils.is_bearish(df)] = -1
        return sig

class BOSFVGFillBull(PatternDetector):
    """BOS + FVG Fill Bullish: a bullish Break of Structure creates a
    Fair Value Gap (candle[i].low > candle[i-2].high during the BOS
    impulse). When price later drops into that FVG = buy entry.

    BOS confirms the trend; FVG marks the institutional re-entry zone."""
    name = "BOS_FVGFillBull"
    category = "multi_structure"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < 50:
            return sig

        swing_high = utils.swing_high_values(df)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        o = df["open"]
        prev_close = close.shift(1)
        body = utils.body_size(df)
        ab = utils.avg_body(df)

        # Detect BOS events (close breaks above swing high with conviction)
        bos = (
            (prev_close <= swing_high.shift(1)) &
            (close > swing_high) &
            (body > ab * 0.8) &
            utils.is_bullish(df) &
            swing_high.notna()
        )

        # Track FVG created during/around BOS bars
        fvg_top = pd.Series(np.nan, index=df.index)
        fvg_bot = pd.Series(np.nan, index=df.index)

        for i in range(2, len(df)):
            if not bos.iloc[i]:
                continue
            # Check if BOS candle created an FVG: low[i] > high[i-2]
            if low.iloc[i] > high.iloc[i - 2]:
                fvg_top.iloc[i] = low.iloc[i]
                fvg_bot.iloc[i] = high.iloc[i - 2]

        fvg_top = fvg_top.ffill()
        fvg_bot = fvg_bot.ffill()

        # Entry when price fills the FVG
        enters = (low <= fvg_top) & (close > fvg_bot) & fvg_top.notna()
        first = enters & ~enters.shift(1).fillna(False).astype(bool)
        first = first & ~bos  # don't signal on the BOS bar

        sig[first & utils.is_bullish(df)] = 1
        return sig

class BOSFVGFillBear(PatternDetector):
    """BOS + FVG Fill Bearish: bearish BOS creates a bearish FVG;
    entry when price rises into the gap."""
    name = "BOS_FVGFillBear"
    category = "multi_structure"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < 50:
            return sig

        swing_low = utils.swing_low_values(df)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        o = df["open"]
        prev_close = close.shift(1)
        body = utils.body_size(df)
        ab = utils.avg_body(df)

        bos = (
            (prev_close >= swing_low.shift(1)) &
            (close < swing_low) &
            (body > ab * 0.8) &
            utils.is_bearish(df) &
            swing_low.notna()
        )

        fvg_top = pd.Series(np.nan, index=df.index)
        fvg_bot = pd.Series(np.nan, index=df.index)

        for i in range(2, len(df)):
            if not bos.iloc[i]:
                continue
            # Bearish FVG: high[i] < low[i-2]
            if high.iloc[i] < low.iloc[i - 2]:
                fvg_top.iloc[i] = low.iloc[i - 2]
                fvg_bot.iloc[i] = high.iloc[i]

        fvg_top = fvg_top.ffill()
        fvg_bot = fvg_bot.ffill()

        enters = (high >= fvg_bot) & (close < fvg_top) & fvg_bot.notna()
        first = enters & ~enters.shift(1).fillna(False).astype(bool)
        first = first & ~bos

        sig[first & utils.is_bearish(df)] = -1
        return sig

class MitigationBlockBull(PatternDetector):
    """Mitigation Block Bullish: a bearish Order Block (last red candle
    before bullish impulse) gets BROKEN to the downside — price trades
    through it, invalidating the demand zone. The zone then flips from
    demand to supply-turned-demand. On first retest FROM ABOVE = buy.

    Conceptually: institutional orders at the OB were "mitigated"
    (absorbed), and the remaining unfilled orders on the other side
    now act as support."""
    name = "MitigationBull"
    category = "multi_structure"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < 50:
            return sig

        o, c, high, low = df["open"], df["close"], df["high"], df["low"]
        atr_val = utils.atr(df)

        # Step 1: Find bullish OBs (last bearish candle before strong up move)
        ob_high = pd.Series(np.nan, index=df.index)
        ob_low = pd.Series(np.nan, index=df.index)
        ob_broken = pd.Series(False, index=df.index)

        for i in range(2, len(df)):
            impulse = (c.iloc[i - 1] - o.iloc[i - 1]) > 2 * atr_val.iloc[i - 1]
            if not impulse:
                continue
            if c.iloc[i - 2] < o.iloc[i - 2]:  # bearish candle = OB
                ob_high.iloc[i - 1] = o.iloc[i - 2]
                ob_low.iloc[i - 1] = c.iloc[i - 2]

        ob_high_ff = ob_high.ffill()
        ob_low_ff = ob_low.ffill()

        # Step 2: Detect when price breaks BELOW the OB (mitigates it)
        mitigation_zone_high = pd.Series(np.nan, index=df.index)
        mitigation_zone_low = pd.Series(np.nan, index=df.index)

        for i in range(1, len(df)):
            oh = ob_high_ff.iloc[i]
            ol = ob_low_ff.iloc[i]
            if pd.isna(oh):
                continue
            # Price closes below the OB low = OB is broken/mitigated
            if c.iloc[i] < ol:
                mitigation_zone_high.iloc[i] = oh
                mitigation_zone_low.iloc[i] = ol

        mit_high = mitigation_zone_high.ffill()
        mit_low = mitigation_zone_low.ffill()

        # Step 3: Retest from above (price comes back up into the zone)
        retest = (
            (low <= mit_high) &
            (c > mit_low) &
            mit_high.notna() &
            utils.is_bullish(df)
        )
        first_retest = retest & ~retest.shift(1).fillna(False).astype(bool)
        sig[first_retest] = 1
        return sig

class MitigationBlockBear(PatternDetector):
    """Mitigation Block Bearish: bullish OB gets broken to upside,
    zone flips from supply to demand-turned-supply. Retest from below = sell."""
    name = "MitigationBear"
    category = "multi_structure"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < 50:
            return sig

        o, c, high, low = df["open"], df["close"], df["high"], df["low"]
        atr_val = utils.atr(df)

        # Find bearish OBs (last bullish candle before strong down move)
        ob_high = pd.Series(np.nan, index=df.index)
        ob_low = pd.Series(np.nan, index=df.index)

        for i in range(2, len(df)):
            impulse = (o.iloc[i - 1] - c.iloc[i - 1]) > 2 * atr_val.iloc[i - 1]
            if not impulse:
                continue
            if c.iloc[i - 2] > o.iloc[i - 2]:  # bullish candle = OB
                ob_high.iloc[i - 1] = c.iloc[i - 2]
                ob_low.iloc[i - 1] = o.iloc[i - 2]

        ob_high_ff = ob_high.ffill()
        ob_low_ff = ob_low.ffill()

        # Price breaks ABOVE the OB high = mitigation
        mitigation_zone_high = pd.Series(np.nan, index=df.index)
        mitigation_zone_low = pd.Series(np.nan, index=df.index)

        for i in range(1, len(df)):
            oh = ob_high_ff.iloc[i]
            ol = ob_low_ff.iloc[i]
            if pd.isna(oh):
                continue
            if c.iloc[i] > oh:
                mitigation_zone_high.iloc[i] = oh
                mitigation_zone_low.iloc[i] = ol

        mit_high = mitigation_zone_high.ffill()
        mit_low = mitigation_zone_low.ffill()

        # Retest from below
        retest = (
            (high >= mit_low) &
            (c < mit_high) &
            mit_low.notna() &
            utils.is_bearish(df)
        )
        first_retest = retest & ~retest.shift(1).fillna(False).astype(bool)
        sig[first_retest] = -1
        return sig

class LiqSweepCHoCHBull(PatternDetector):
    """Liquidity Sweep + CHoCH Bullish: price sweeps below a confirmed
    swing low (wick below, close above = stop hunt), and then within
    3 bars a bullish CHoCH occurs. The sweep collects liquidity; the
    CHoCH confirms the reversal. Extremely high probability setup."""
    name = "LiqSweep_CHoCHBull"
    category = "multi_structure"

    LOOKBACK = 3  # max bars between sweep and CHoCH

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < 50:
            return sig

        swing_low = utils.swing_low_values(df)
        low = df["low"]
        close = df["close"]

        # Liquidity sweep: wick below swing low, close above it
        sweep = (low < swing_low) & (close > swing_low) & swing_low.notna()

        # Bullish CHoCH events
        bull_choch, _ = _choch_events(df)

        # Check if a sweep happened within LOOKBACK bars before CHoCH
        sweep_recently = sweep.rolling(self.LOOKBACK + 1, min_periods=1).max()

        # Signal: bullish CHoCH AND sweep happened within lookback
        combo = bull_choch & (sweep_recently > 0)
        sig[combo] = 1
        return sig

class LiqSweepCHoCHBear(PatternDetector):
    """Liquidity Sweep + CHoCH Bearish: sweep above swing high
    (stop hunt of longs), then bearish CHoCH within 3 bars."""
    name = "LiqSweep_CHoCHBear"
    category = "multi_structure"

    LOOKBACK = 3

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        if len(df) < 50:
            return sig

        swing_high = utils.swing_high_values(df)
        high = df["high"]
        close = df["close"]

        # Liquidity sweep: wick above swing high, close below
        sweep = (high > swing_high) & (close < swing_high) & swing_high.notna()

        _, bear_choch = _choch_events(df)

        sweep_recently = sweep.rolling(self.LOOKBACK + 1, min_periods=1).max()

        combo = bear_choch & (sweep_recently > 0)
        sig[combo] = -1
        return sig

ALL_MULTI_STRUCTURE = [
    OneTwoReversalBull, OneTwoReversalBear,
    CHoCHOBRetestBull, CHoCHOBRetestBear,
    BOSFVGFillBull, BOSFVGFillBear,
    MitigationBlockBull, MitigationBlockBear,
    LiqSweepCHoCHBull, LiqSweepCHoCHBear,
]
