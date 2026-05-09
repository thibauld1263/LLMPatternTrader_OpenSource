"""
PatternTrader - Parametric Candlestick Pattern Engine (no look-ahead)
Data-driven pattern detector: each pattern is defined by a config dict
specifying body size, wick position, ATR gates, candle type, and
inter-candle boolean conditions across up to 3 candles.

17 patterns defined from a structured specification table.
"""

import pandas as pd
import numpy as np
from patterns.base import PatternDetector
import ta_utils as utils


# ─────────────────────────────────────────────────────────────────────
#  Pattern Configuration Table
# ─────────────────────────────────────────────────────────────────────
# Each config is a dict with keys for candle 1, 2, 3 parameters.
# Boolean flags: 1 = condition must be true, 0 = skip.
# Body/position values: percentage of candle range.
# ATR_Multiplier: minimum range as a multiple of ATR (size gate).
# Body_PositionTop/Bot: max upper/lower wick as % of range.
#   0 = don't check, 100 = unconstrained (always passes).

PATTERN_CONFIGS = [
    # ── 1. Bearish Gravestone Doji ────────────────────────────────
    {
        "name": "P_BearGravestoneDoji",
        "reversal": "bearish",
        "c1": {"min_body": 0, "max_body": 15, "atr_mult": 1, "type": "bullish",
               "pos_top": 100, "pos_bot": 25},
        "use_c2": False, "c2": {}, "c2_conds": {},
        "use_c3": False, "c3": {}, "c3_c2_conds": {}, "c3_c1_conds": {},
    },
    # ── 2. Bearish Strong Gravestone Doji ─────────────────────────
    {
        "name": "P_BearStrongGravestoneDoji",
        "reversal": "bearish",
        "c1": {"min_body": 0, "max_body": 15, "atr_mult": 1, "type": "bearish",
               "pos_top": 100, "pos_bot": 25},
        "use_c2": False, "c2": {}, "c2_conds": {},
        "use_c3": False, "c3": {}, "c3_c2_conds": {}, "c3_c1_conds": {},
    },
    # ── 3. Bullish Dragonfly Doji ─────────────────────────────────
    {
        "name": "P_BullDragonflyDoji",
        "reversal": "bullish",
        "c1": {"min_body": 0, "max_body": 15, "atr_mult": 1, "type": "bearish",
               "pos_top": 25, "pos_bot": 100},
        "use_c2": False, "c2": {}, "c2_conds": {},
        "use_c3": False, "c3": {}, "c3_c2_conds": {}, "c3_c1_conds": {},
    },
    # ── 4. Bullish Strong Dragonfly Doji ──────────────────────────
    {
        "name": "P_BullStrongDragonflyDoji",
        "reversal": "bullish",
        "c1": {"min_body": 0, "max_body": 15, "atr_mult": 1, "type": "bullish",
               "pos_top": 25, "pos_bot": 100},
        "use_c2": False, "c2": {}, "c2_conds": {},
        "use_c3": False, "c3": {}, "c3_c2_conds": {}, "c3_c1_conds": {},
    },
    # ── 5. Hanging Man ────────────────────────────────────────────
    {
        "name": "P_HangingMan",
        "reversal": "bearish",
        "c1": {"min_body": 30, "max_body": 50, "atr_mult": 1, "type": "bullish",
               "pos_top": 50, "pos_bot": 100},
        "use_c2": False, "c2": {}, "c2_conds": {},
        "use_c3": False, "c3": {}, "c3_c2_conds": {}, "c3_c1_conds": {},
    },
    # ── 6. Hanging Man (Strong) ───────────────────────────────────
    {
        "name": "P_HangingManStrong",
        "reversal": "bearish",
        "c1": {"min_body": 30, "max_body": 50, "atr_mult": 1, "type": "bearish",
               "pos_top": 50, "pos_bot": 100},
        "use_c2": False, "c2": {}, "c2_conds": {},
        "use_c3": False, "c3": {}, "c3_c2_conds": {}, "c3_c1_conds": {},
    },
    # ── 7. Shooting Star ──────────────────────────────────────────
    {
        "name": "P_ShootingStar",
        "reversal": "bearish",
        "c1": {"min_body": 0, "max_body": 30, "atr_mult": 1, "type": "bullish",
               "pos_top": 100, "pos_bot": 40},
        "use_c2": False, "c2": {}, "c2_conds": {},
        "use_c3": False, "c3": {}, "c3_c2_conds": {}, "c3_c1_conds": {},
    },
    # ── 8. Shooting Star Strong ───────────────────────────────────
    {
        "name": "P_ShootingStarStrong",
        "reversal": "bearish",
        "c1": {"min_body": 0, "max_body": 30, "atr_mult": 1, "type": "bearish",
               "pos_top": 100, "pos_bot": 40},
        "use_c2": False, "c2": {}, "c2_conds": {},
        "use_c3": False, "c3": {}, "c3_c2_conds": {}, "c3_c1_conds": {},
    },
    # ── 9. Bullish Hammer ─────────────────────────────────────────
    {
        "name": "P_BullHammer",
        "reversal": "bullish",
        "c1": {"min_body": 30, "max_body": 50, "atr_mult": 1, "type": "bullish",
               "pos_top": 50, "pos_bot": 100},
        "use_c2": False, "c2": {}, "c2_conds": {},
        "use_c3": False, "c3": {}, "c3_c2_conds": {}, "c3_c1_conds": {},
    },
    # ── 10. Inverted Bullish Hammer ───────────────────────────────
    {
        "name": "P_InvBullHammer",
        "reversal": "bullish",
        "c1": {"min_body": 30, "max_body": 50, "atr_mult": 1, "type": "bullish",
               "pos_top": 100, "pos_bot": 50},
        "use_c2": False, "c2": {}, "c2_conds": {},
        "use_c3": False, "c3": {}, "c3_c2_conds": {}, "c3_c1_conds": {},
    },
    # ── 11. Three Black Crows ─────────────────────────────────────
    {
        "name": "P_ThreeBlackCrows",
        "reversal": "bearish",
        "c1": {"min_body": 50, "max_body": 100, "atr_mult": 0.6, "type": "bearish",
               "pos_top": 0, "pos_bot": 0},
        "use_c2": True,
        "c2": {"min_body": 70, "max_body": 100, "atr_mult": 0.6, "type": "bearish",
               "pos_top": 0, "pos_bot": 0},
        "c2_conds": {"LLL": 1, "CLL": 1, "CLO": 1, "CLC": 1},
        "use_c3": True,
        "c3": {"min_body": 50, "max_body": 100, "atr_mult": 0.6, "type": "bearish",
               "pos_top": 0, "pos_bot": 0},
        "c3_c2_conds": {"LLL": 1, "CLL": 1, "CLO": 1, "CLC": 1},
        "c3_c1_conds": {"LLL": 1, "CLL": 1, "CLO": 1, "CLC": 1},
    },
    # ── 12. Strong Bullish Engulfing ──────────────────────────────
    {
        "name": "P_StrongBullEngulfing",
        "reversal": "bullish",
        "c1": {"min_body": 50, "max_body": 100, "atr_mult": 0.6, "type": "bearish",
               "pos_top": 0, "pos_bot": 0},
        "use_c2": True,
        "c2": {"min_body": 70, "max_body": 100, "atr_mult": 1, "type": "bullish",
               "pos_top": 0, "pos_bot": 0},
        "c2_conds": {"HHH": 1, "CHH": 1, "CHO": 1, "CHC": 1},
        "use_c3": False, "c3": {}, "c3_c2_conds": {}, "c3_c1_conds": {},
    },
    # ── 13. Strong Bearish Engulfing ──────────────────────────────
    {
        "name": "P_StrongBearEngulfing",
        "reversal": "bearish",
        "c1": {"min_body": 50, "max_body": 100, "atr_mult": 0.6, "type": "bullish",
               "pos_top": 0, "pos_bot": 0},
        "use_c2": True,
        "c2": {"min_body": 70, "max_body": 100, "atr_mult": 1, "type": "bearish",
               "pos_top": 0, "pos_bot": 0},
        "c2_conds": {"LLL": 1, "CLL": 1, "CLO": 1, "CLC": 1},
        "use_c3": False, "c3": {}, "c3_c2_conds": {}, "c3_c1_conds": {},
    },
    # ── 14. Evening Star ──────────────────────────────────────────
    {
        "name": "P_EveningStar",
        "reversal": "bearish",
        "c1": {"min_body": 70, "max_body": 100, "atr_mult": 1, "type": "bullish",
               "pos_top": 0, "pos_bot": 0},
        "use_c2": True,
        "c2": {"min_body": 0, "max_body": 30, "atr_mult": 0.5, "type": "bearish",
               "pos_top": 65, "pos_bot": 65},
        "c2_conds": {"HHH": 1, "CHO": 1, "CLC": 1},
        "use_c3": True,
        "c3": {"min_body": 80, "max_body": 100, "atr_mult": 1, "type": "bearish",
               "pos_top": 0, "pos_bot": 0},
        "c3_c2_conds": {"LLL": 1, "CLL": 1, "CLO": 1, "CLC": 1},
        "c3_c1_conds": {"CLO": 1, "CLC": 1},
    },
    # ── 15. Evening Star 2 (bullish star candle) ──────────────────
    {
        "name": "P_EveningStar2",
        "reversal": "bearish",
        "c1": {"min_body": 70, "max_body": 100, "atr_mult": 1, "type": "bullish",
               "pos_top": 0, "pos_bot": 0},
        "use_c2": True,
        "c2": {"min_body": 0, "max_body": 30, "atr_mult": 0.5, "type": "bullish",
               "pos_top": 65, "pos_bot": 65},
        "c2_conds": {"HHH": 1, "CHO": 1, "CHC": 1},
        "use_c3": True,
        "c3": {"min_body": 80, "max_body": 100, "atr_mult": 1, "type": "bearish",
               "pos_top": 0, "pos_bot": 0},
        "c3_c2_conds": {"LLL": 1, "CLL": 1, "CLO": 1, "CLC": 1},
        "c3_c1_conds": {"CLO": 1, "CLC": 1},
    },
    # ── 16. Shooting Star w/ Confirmation ─────────────────────────
    {
        "name": "P_ShootingStarConf",
        "reversal": "bearish",
        "c1": {"min_body": 0, "max_body": 30, "atr_mult": 1, "type": "bullish",
               "pos_top": 100, "pos_bot": 40},
        "use_c2": True,
        "c2": {"min_body": 0, "max_body": 100, "atr_mult": 0, "type": "bearish",
               "pos_top": 0, "pos_bot": 0},
        "c2_conds": {},
        "use_c3": False, "c3": {}, "c3_c2_conds": {}, "c3_c1_conds": {},
    },
    # ── 17. Shooting Star Strong w/ Confirmation ──────────────────
    {
        "name": "P_ShootingStarStrongConf",
        "reversal": "bearish",
        "c1": {"min_body": 0, "max_body": 30, "atr_mult": 1, "type": "bearish",
               "pos_top": 100, "pos_bot": 40},
        "use_c2": True,
        "c2": {"min_body": 0, "max_body": 100, "atr_mult": 0, "type": "bearish",
               "pos_top": 0, "pos_bot": 0},
        "c2_conds": {},
        "use_c3": False, "c3": {}, "c3_c2_conds": {}, "c3_c1_conds": {},
    },
]


# ─────────────────────────────────────────────────────────────────────
#  Generic Parametric Engine
# ─────────────────────────────────────────────────────────────────────

def _check_candle(o, c, h, l, rng, atr_val, cfg):
    """
    Check a single candle against its config.
    Returns a boolean Series: True where the candle matches all constraints.
    """
    body = (c - o).abs()
    # Avoid division by zero
    rng_safe = rng.replace(0, np.nan)
    body_pct = body / rng_safe * 100

    # Body size bounds
    match = (body_pct >= cfg["min_body"]) & (body_pct <= cfg["max_body"])

    # ATR size gate: candle range must be >= atr_mult * ATR
    if cfg["atr_mult"] > 0:
        match = match & (rng >= cfg["atr_mult"] * atr_val)

    # Candle type
    if cfg["type"] == "bullish":
        match = match & (c > o)
    elif cfg["type"] == "bearish":
        match = match & (c < o)

    # Body position constraints (wick size limits)
    body_top = pd.concat([o, c], axis=1).max(axis=1)
    body_bot = pd.concat([o, c], axis=1).min(axis=1)
    upper_wick_pct = (h - body_top) / rng_safe * 100
    lower_wick_pct = (body_bot - l) / rng_safe * 100

    if cfg["pos_top"] > 0:
        match = match & (upper_wick_pct <= cfg["pos_top"])
    if cfg["pos_bot"] > 0:
        match = match & (lower_wick_pct <= cfg["pos_bot"])

    return match.fillna(False)


def _check_inter_candle(conds, later_o, later_c, later_h, later_l,
                         earlier_o, earlier_c, earlier_h, earlier_l,
                         atr_val):
    """
    Apply inter-candle boolean conditions.
    Returns a boolean Series: True where all required conditions hold.
    """
    match = pd.Series(True, index=later_o.index)

    flag_map = {
        "HHH": lambda: later_h > earlier_h,
        "LLL": lambda: later_l < earlier_l,
        "CHH": lambda: later_c > earlier_h,
        "CLL": lambda: later_c < earlier_l,
        "CLO": lambda: later_c < earlier_o,
        "CHO": lambda: later_c > earlier_o,
        "CHC": lambda: later_c > earlier_c,
        "CLC": lambda: later_c < earlier_c,
        "OLC": lambda: later_o < earlier_c,
        "OHC": lambda: later_o > earlier_c,
    }

    for flag, check_fn in flag_map.items():
        if conds.get(flag, 0) == 1:
            match = match & check_fn()

    # ATR proximity checks (value = X multiplier, 0 = skip)
    prox_map = {
        "OXC": (later_o, earlier_c),
        "CXO": (later_c, earlier_o),
        "LXL": (later_l, earlier_l),
        "HXH": (later_h, earlier_h),
    }
    for flag, (a, b) in prox_map.items():
        x = conds.get(flag, 0)
        if x > 0:
            match = match & ((a - b).abs() <= x * atr_val)

    return match


class ParametricCandlePattern(PatternDetector):
    """
    Generic data-driven candlestick pattern detector.
    Takes a config dict and evaluates body size, body position,
    ATR gates, candle type, and inter-candle conditions.
    """
    category = "parametric_candle"

    def __init__(self, config: dict):
        self.config = config
        self.name = config["name"]
        self._reversal = config["reversal"]
        self._signal_val = 1 if self._reversal == "bullish" else -1

    def detect(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)

        n_candles = 1
        if self.config.get("use_c3"):
            n_candles = 3
        elif self.config.get("use_c2"):
            n_candles = 2

        if len(df) < max(n_candles + 1, 20):
            return sig

        atr_val = utils.atr(df)
        o, c, h, l = df["open"], df["close"], df["high"], df["low"]
        rng = h - l

        # Determine shifts: C1 is oldest, last candle is current bar
        if n_candles == 1:
            c1_shift = 0
        elif n_candles == 2:
            c1_shift = 1
        else:  # 3
            c1_shift = 2

        # ── Candle 1 ──
        c1_o = o.shift(c1_shift)
        c1_c = c.shift(c1_shift)
        c1_h = h.shift(c1_shift)
        c1_l = l.shift(c1_shift)
        c1_rng = rng.shift(c1_shift)
        c1_atr = atr_val.shift(c1_shift)

        match = _check_candle(c1_o, c1_c, c1_h, c1_l, c1_rng, c1_atr,
                              self.config["c1"])

        # ── Candle 2 ──
        if self.config.get("use_c2"):
            if n_candles == 2:
                c2_shift = 0  # current bar
            else:  # 3 candles
                c2_shift = 1

            c2_o = o.shift(c2_shift)
            c2_c = c.shift(c2_shift)
            c2_h = h.shift(c2_shift)
            c2_l = l.shift(c2_shift)
            c2_rng = rng.shift(c2_shift)
            c2_atr = atr_val.shift(c2_shift)

            match = match & _check_candle(
                c2_o, c2_c, c2_h, c2_l, c2_rng, c2_atr,
                self.config["c2"]
            )
            match = match & _check_inter_candle(
                self.config["c2_conds"],
                c2_o, c2_c, c2_h, c2_l,
                c1_o, c1_c, c1_h, c1_l,
                c1_atr
            )

        # ── Candle 3 ──
        if self.config.get("use_c3"):
            c3_o = o  # current bar (shift=0)
            c3_c = c
            c3_h = h
            c3_l = l
            c3_rng = rng
            c3_atr = atr_val

            match = match & _check_candle(
                c3_o, c3_c, c3_h, c3_l, c3_rng, c3_atr,
                self.config["c3"]
            )
            # C3 vs C2 conditions
            match = match & _check_inter_candle(
                self.config["c3_c2_conds"],
                c3_o, c3_c, c3_h, c3_l,
                c2_o, c2_c, c2_h, c2_l,
                c2_atr
            )
            # C3 vs C1 conditions
            match = match & _check_inter_candle(
                self.config["c3_c1_conds"],
                c3_o, c3_c, c3_h, c3_l,
                c1_o, c1_c, c1_h, c1_l,
                c1_atr
            )

        sig[match] = self._signal_val
        return sig

    def __repr__(self):
        return f"<{self.category}.{self.name}>"


# ─────────────────────────────────────────────────────────────────────
#  Auto-generate detector instances from configs
# ─────────────────────────────────────────────────────────────────────

ALL_PARAMETRIC_CANDLES = [ParametricCandlePattern(cfg) for cfg in PATTERN_CONFIGS]
