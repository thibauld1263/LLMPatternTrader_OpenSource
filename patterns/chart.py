"""
PatternTrader - Chart Patterns
8 patterns: DoubleTop, DoubleBottom, TripleTop, TripleBottom,
HeadAndShoulders, InvHeadAndShoulders, RisingWedge, FallingWedge
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils

class DoubleTop(PatternDetector):
    name = "DoubleTop"
    category = "chart"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        sh = utils.swing_high_values(df)
        close = df["close"]
        atr_val = utils.atr(df)
        prev_sh = sh.shift(20)
        tol = atr_val * 0.5
        two_peaks = (sh - prev_sh).abs() < tol
        breakdown = close < close.shift(1)
        sig[two_peaks & breakdown & sh.notna() & prev_sh.notna()] = -1
        return sig

class DoubleBottom(PatternDetector):
    name = "DoubleBottom"
    category = "chart"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        sl = utils.swing_low_values(df)
        close = df["close"]
        atr_val = utils.atr(df)
        prev_sl = sl.shift(20)
        tol = atr_val * 0.5
        two_troughs = (sl - prev_sl).abs() < tol
        breakup = close > close.shift(1)
        sig[two_troughs & breakup & sl.notna() & prev_sl.notna()] = 1
        return sig

class TripleTop(PatternDetector):
    name = "TripleTop"
    category = "chart"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        sh = utils.swing_high_values(df)
        atr_val = utils.atr(df)
        tol = atr_val * 0.5
        p1 = sh.shift(40)
        p2 = sh.shift(20)
        p3 = sh
        close = df["close"]
        mask = (
            ((p1 - p2).abs() < tol) &
            ((p2 - p3).abs() < tol) &
            (close < close.shift(1)) &
            p1.notna() & p2.notna() & p3.notna()
        )
        sig[mask] = -1
        return sig

class TripleBottom(PatternDetector):
    name = "TripleBottom"
    category = "chart"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        sl = utils.swing_low_values(df)
        atr_val = utils.atr(df)
        tol = atr_val * 0.5
        t1 = sl.shift(40)
        t2 = sl.shift(20)
        t3 = sl
        close = df["close"]
        mask = (
            ((t1 - t2).abs() < tol) &
            ((t2 - t3).abs() < tol) &
            (close > close.shift(1)) &
            t1.notna() & t2.notna() & t3.notna()
        )
        sig[mask] = 1
        return sig

class HeadAndShoulders(PatternDetector):
    name = "HeadAndShoulders"
    category = "chart"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        sh = utils.swing_high_values(df)
        atr_val = utils.atr(df)
        tol = atr_val * 0.5
        left = sh.shift(40)
        head = sh.shift(20)
        right = sh
        close = df["close"]
        mask = (
            (head > left + tol) &
            (head > right + tol) &
            ((left - right).abs() < tol) &
            (close < close.shift(1)) &
            left.notna() & head.notna() & right.notna()
        )
        sig[mask] = -1
        return sig

class InvHeadAndShoulders(PatternDetector):
    name = "InvHeadAndShoulders"
    category = "chart"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        sl = utils.swing_low_values(df)
        atr_val = utils.atr(df)
        tol = atr_val * 0.5
        left = sl.shift(40)
        head = sl.shift(20)
        right = sl
        close = df["close"]
        mask = (
            (head < left - tol) &
            (head < right - tol) &
            ((left - right).abs() < tol) &
            (close > close.shift(1)) &
            left.notna() & head.notna() & right.notna()
        )
        sig[mask] = 1
        return sig

class RisingWedge(PatternDetector):
    name = "RisingWedge"
    category = "chart"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        period = 30
        close = df["close"]
        high = df["high"]
        low = df["low"]
        high_slope = (high - high.shift(period)) / period
        low_slope = (low - low.shift(period)) / period
        mask = (
            (high_slope > 0) &
            (low_slope > 0) &
            (high_slope < low_slope) &
            (close < close.shift(1))
        )
        sig[mask] = -1
        return sig

class FallingWedge(PatternDetector):
    name = "FallingWedge"
    category = "chart"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        period = 30
        close = df["close"]
        high = df["high"]
        low = df["low"]
        high_slope = (high - high.shift(period)) / period
        low_slope = (low - low.shift(period)) / period
        mask = (
            (high_slope < 0) &
            (low_slope < 0) &
            (high_slope > low_slope) &
            (close > close.shift(1))
        )
        sig[mask] = 1
        return sig

ALL_CHART = [
    DoubleTop, DoubleBottom, TripleTop, TripleBottom,
    HeadAndShoulders, InvHeadAndShoulders,
    RisingWedge, FallingWedge,
]
