"""
PatternTrader - Pattern Registry
Auto-registers all pattern detectors.
"""

import pandas as pd
from typing import List
from patterns.base import PatternDetector

# Original 36
from patterns.candlestick import ALL_CANDLESTICK
from patterns.chart import ALL_CHART
from patterns.trend import ALL_TREND
from patterns.momentum import ALL_MOMENTUM
from patterns.volatility import ALL_VOLATILITY

# Institutional / Smart Money (32)
from patterns.liquidity import ALL_LIQUIDITY
from patterns.order_flow import ALL_ORDER_FLOW
from patterns.smart_money import ALL_SMART_MONEY
from patterns.session import ALL_SESSION
from patterns.institutional import ALL_INSTITUTIONAL

# Advanced (10)
from patterns.advanced import ALL_ADVANCED

# Advanced Confluence (7)
from patterns.advanced_confluence import ALL_ADVANCED_CONFLUENCE

# Structural (12)
from patterns.structural import ALL_STRUCTURAL

# Harmonics (4)
from patterns.harmonics import ALL_HARMONICS

# Multi-Structure Combos (10)
from patterns.multi_structure import ALL_MULTI_STRUCTURE

# Parametric Candles (17)
from patterns.parametric_candles import ALL_PARAMETRIC_CANDLES

# Bulkowski Chart Patterns (8)
from patterns.bulkowski_patterns import ALL_BULKOWSKI

# Complex Confluence (10)
from patterns.complex_confluence import ALL_CONFLUENCE

class PatternRegistry:

    def __init__(self):
        self.detectors: List[PatternDetector] = []
        for cls_list in [
            ALL_CANDLESTICK, ALL_CHART, ALL_TREND, ALL_MOMENTUM, ALL_VOLATILITY,
            ALL_LIQUIDITY, ALL_ORDER_FLOW, ALL_SMART_MONEY, ALL_SESSION,
            ALL_INSTITUTIONAL, ALL_ADVANCED, ALL_STRUCTURAL, ALL_HARMONICS,
            ALL_ADVANCED_CONFLUENCE, ALL_MULTI_STRUCTURE,
            ALL_PARAMETRIC_CANDLES, ALL_BULKOWSKI, ALL_CONFLUENCE
        ]:
            for item in cls_list:
                # Support both classes (instantiate) and pre-built instances
                if isinstance(item, PatternDetector):
                    self.detectors.append(item)
                else:
                    self.detectors.append(item())

    @property
    def count(self) -> int:
        return len(self.detectors)

    def scan(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = {}
        for det in self.detectors:
            try:
                signals[det.name] = det.detect(df)
            except Exception:
                signals[det.name] = pd.Series(0, index=df.index)
        return pd.DataFrame(signals, index=df.index)

    def scan_specific(self, df: pd.DataFrame, pattern_names: list) -> pd.DataFrame:
        """Scan ONLY the specified patterns — used for fast subsequent runs."""
        name_set = set(pattern_names)
        signals = {}
        for det in self.detectors:
            if det.name in name_set:
                try:
                    signals[det.name] = det.detect(df)
                except Exception:
                    signals[det.name] = pd.Series(0, index=df.index)
        return pd.DataFrame(signals, index=df.index)

    def summary(self) -> pd.DataFrame:
        rows = [{"name": d.name, "category": d.category} for d in self.detectors]
        return pd.DataFrame(rows)
