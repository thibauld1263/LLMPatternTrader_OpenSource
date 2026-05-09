"""
PatternTrader - Candlestick Patterns
13 patterns: Hammer, InvertedHammer, BullishEngulfing, BearishEngulfing,
MorningStar, EveningStar, ThreeWhiteSoldiers, ThreeBlackCrows,
Doji, DarkCloudCover, PiercingLine, ShootingStar, HangingMan
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils

class Hammer(PatternDetector):
    name = "Hammer"
    category = "candlestick"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        body = utils.body_size(df)
        ls = utils.lower_shadow(df)
        us = utils.upper_shadow(df)
        rng = utils.candle_range(df)
        ab = utils.avg_body(df)
        mask = (
            (ls >= 2 * body) &
            (us <= body * 0.3) &
            (body > 0) &
            (body < ab * 0.8) &
            (rng > 0)
        )
        sig[mask] = 1
        return sig

class InvertedHammer(PatternDetector):
    name = "InvertedHammer"
    category = "candlestick"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        body = utils.body_size(df)
        ls = utils.lower_shadow(df)
        us = utils.upper_shadow(df)
        ab = utils.avg_body(df)
        mask = (
            (us >= 2 * body) &
            (ls <= body * 0.3) &
            (body > 0) &
            (body < ab * 0.8)
        )
        sig[mask] = 1
        return sig

class ShootingStar(PatternDetector):
    name = "ShootingStar"
    category = "candlestick"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        body = utils.body_size(df)
        us = utils.upper_shadow(df)
        ls = utils.lower_shadow(df)
        ab = utils.avg_body(df)
        mask = (
            (us >= 2 * body) &
            (ls <= body * 0.3) &
            (body > 0) &
            (body < ab * 0.8) &
            utils.is_bearish(df)
        )
        sig[mask] = -1
        return sig

class HangingMan(PatternDetector):
    name = "HangingMan"
    category = "candlestick"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        body = utils.body_size(df)
        ls = utils.lower_shadow(df)
        us = utils.upper_shadow(df)
        ab = utils.avg_body(df)
        mask = (
            (ls >= 2 * body) &
            (us <= body * 0.3) &
            (body > 0) &
            (body < ab * 0.8) &
            utils.is_bearish(df)
        )
        sig[mask] = -1
        return sig

class BullishEngulfing(PatternDetector):
    name = "BullishEngulfing"
    category = "candlestick"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        o, c = df["open"], df["close"]
        prev_o, prev_c = o.shift(1), c.shift(1)
        mask = (
            (prev_c < prev_o) &  # prev bearish
            (c > o) &            # current bullish
            (o <= prev_c) &      # open below prev close
            (c >= prev_o)        # close above prev open
        )
        sig[mask] = 1
        return sig

class BearishEngulfing(PatternDetector):
    name = "BearishEngulfing"
    category = "candlestick"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        o, c = df["open"], df["close"]
        prev_o, prev_c = o.shift(1), c.shift(1)
        mask = (
            (prev_c > prev_o) &
            (c < o) &
            (o >= prev_c) &
            (c <= prev_o)
        )
        sig[mask] = -1
        return sig

class MorningStar(PatternDetector):
    name = "MorningStar"
    category = "candlestick"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        o, c = df["open"], df["close"]
        body = utils.body_size(df)
        ab = utils.avg_body(df)
        for i in range(2, len(df)):
            b1_bear = c.iloc[i-2] < o.iloc[i-2] and body.iloc[i-2] > ab.iloc[i-2] * 0.5
            b2_small = body.iloc[i-1] < ab.iloc[i-1] * 0.3
            b3_bull = c.iloc[i] > o.iloc[i] and body.iloc[i] > ab.iloc[i] * 0.5
            b3_close_above = c.iloc[i] > (o.iloc[i-2] + c.iloc[i-2]) / 2
            if b1_bear and b2_small and b3_bull and b3_close_above:
                sig.iloc[i] = 1
        return sig

class EveningStar(PatternDetector):
    name = "EveningStar"
    category = "candlestick"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        o, c = df["open"], df["close"]
        body = utils.body_size(df)
        ab = utils.avg_body(df)
        for i in range(2, len(df)):
            b1_bull = c.iloc[i-2] > o.iloc[i-2] and body.iloc[i-2] > ab.iloc[i-2] * 0.5
            b2_small = body.iloc[i-1] < ab.iloc[i-1] * 0.3
            b3_bear = c.iloc[i] < o.iloc[i] and body.iloc[i] > ab.iloc[i] * 0.5
            b3_close_below = c.iloc[i] < (o.iloc[i-2] + c.iloc[i-2]) / 2
            if b1_bull and b2_small and b3_bear and b3_close_below:
                sig.iloc[i] = -1
        return sig

class ThreeWhiteSoldiers(PatternDetector):
    name = "ThreeWhiteSoldiers"
    category = "candlestick"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        o, c = df["open"], df["close"]
        body = utils.body_size(df)
        ab = utils.avg_body(df)
        for i in range(2, len(df)):
            all_bull = all(c.iloc[i-j] > o.iloc[i-j] for j in range(3))
            rising = c.iloc[i] > c.iloc[i-1] > c.iloc[i-2]
            decent_body = all(body.iloc[i-j] > ab.iloc[i-j] * 0.5 for j in range(3))
            if all_bull and rising and decent_body:
                sig.iloc[i] = 1
        return sig

class ThreeBlackCrows(PatternDetector):
    name = "ThreeBlackCrows"
    category = "candlestick"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        o, c = df["open"], df["close"]
        body = utils.body_size(df)
        ab = utils.avg_body(df)
        for i in range(2, len(df)):
            all_bear = all(c.iloc[i-j] < o.iloc[i-j] for j in range(3))
            falling = c.iloc[i] < c.iloc[i-1] < c.iloc[i-2]
            decent_body = all(body.iloc[i-j] > ab.iloc[i-j] * 0.5 for j in range(3))
            if all_bear and falling and decent_body:
                sig.iloc[i] = -1
        return sig

class Doji(PatternDetector):
    name = "Doji"
    category = "candlestick"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        body = utils.body_size(df)
        rng = utils.candle_range(df)
        ab = utils.avg_body(df)
        mask = (body < rng * 0.1) & (rng > ab * 0.3)
        # Check prior trend
        close_5 = df["close"].rolling(5).mean()
        bullish_prior = df["close"].shift(1) < close_5.shift(1)
        bearish_prior = df["close"].shift(1) > close_5.shift(1)
        sig[mask & bullish_prior] = 1
        sig[mask & bearish_prior] = -1
        return sig

class DarkCloudCover(PatternDetector):
    name = "DarkCloudCover"
    category = "candlestick"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        o, c, h = df["open"], df["close"], df["high"]
        prev_o, prev_c = o.shift(1), c.shift(1)
        midpoint = (prev_o + prev_c) / 2
        mask = (
            (prev_c > prev_o) &
            (o > prev_c) &
            (c < o) &
            (c < midpoint) &
            (c > prev_o)
        )
        sig[mask] = -1
        return sig

class PiercingLine(PatternDetector):
    name = "PiercingLine"
    category = "candlestick"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        o, c = df["open"], df["close"]
        prev_o, prev_c = o.shift(1), c.shift(1)
        midpoint = (prev_o + prev_c) / 2
        mask = (
            (prev_c < prev_o) &
            (o < prev_c) &
            (c > o) &
            (c > midpoint) &
            (c < prev_o)
        )
        sig[mask] = 1
        return sig

# ── Registry list ────────────────────────────────────────────────────

ALL_CANDLESTICK = [
    Hammer, InvertedHammer, ShootingStar, HangingMan,
    BullishEngulfing, BearishEngulfing,
    MorningStar, EveningStar,
    ThreeWhiteSoldiers, ThreeBlackCrows,
    Doji, DarkCloudCover, PiercingLine,
]
