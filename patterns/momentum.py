"""
PatternTrader - Momentum Patterns
5 patterns: RSI_Overbought, RSI_Oversold, RSI_BullDiv, RSI_BearDiv, MACD_HistDiv
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils
import config

class RSIOverbought(PatternDetector):
    name = "RSI_Overbought"
    category = "momentum"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        rsi_val = utils.rsi(df)
        prev_rsi = rsi_val.shift(1)
        sig[(prev_rsi >= 70) & (rsi_val < 70)] = -1
        return sig

class RSIOversold(PatternDetector):
    name = "RSI_Oversold"
    category = "momentum"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        rsi_val = utils.rsi(df)
        prev_rsi = rsi_val.shift(1)
        sig[(prev_rsi <= 30) & (rsi_val > 30)] = 1
        return sig

class RSIBullDiv(PatternDetector):
    name = "RSI_BullDiv"
    category = "momentum"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        rsi_val = utils.rsi(df)
        close = df["close"]
        lookback = 14
        price_lower = close < close.rolling(lookback).min().shift(1)
        rsi_higher = rsi_val > rsi_val.rolling(lookback).min().shift(1)
        sig[price_lower & rsi_higher & (rsi_val < 40)] = 1
        return sig

class RSIBearDiv(PatternDetector):
    name = "RSI_BearDiv"
    category = "momentum"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        rsi_val = utils.rsi(df)
        close = df["close"]
        lookback = 14
        price_higher = close > close.rolling(lookback).max().shift(1)
        rsi_lower = rsi_val < rsi_val.rolling(lookback).max().shift(1)
        sig[price_higher & rsi_lower & (rsi_val > 60)] = -1
        return sig

class MACDHistDiv(PatternDetector):
    name = "MACD_HistDiv"
    category = "momentum"

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        _, _, hist = utils.macd(df)
        prev_hist = hist.shift(1)
        # Bullish: histogram crosses from negative to positive
        sig[(prev_hist < 0) & (hist >= 0)] = 1
        # Bearish: histogram crosses from positive to negative
        sig[(prev_hist > 0) & (hist <= 0)] = -1
        return sig

ALL_MOMENTUM = [
    RSIOverbought, RSIOversold,
    RSIBullDiv, RSIBearDiv,
    MACDHistDiv,
]
