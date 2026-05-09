"""
PatternTrader - Configuration
"""
from datetime import datetime

# MT5 timeframe constants
TIMEFRAME_M5  = 5
TIMEFRAME_M15 = 15
TIMEFRAME_M30 = 30
TIMEFRAME_H1  = 16385
TIMEFRAME_H4  = 16388
TIMEFRAME_D1  = 16408

# ── MT5 ──────────────────────────────────────────────────────────────────
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# ── Symbols ──────────────────────────────────────────────────────────────
SYMBOLS = ["AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY",
            "CHFJPY", "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURUSD",
            "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD", "NZDCAD", "NZDCHF", "NZDJPY",
            "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]

# ── Data Start Date ──────────────────────────────────────────────────────
# All backtesting fetches data from this date instead of a fixed bar count.
DATA_START_DATE = datetime(2020, 1, 1)

# ── Timeframes (per-TF SL/TP/hold) ──────────────────────────────────────
TIMEFRAMES = {

    "M5": {
        "tf_const": TIMEFRAME_M5,
        "sl_mult":  2.5,
        "tp_mult":  3.0,
        "hold":     15,
        "minutes":  5,
    },
    
    "M15": {
        "tf_const": TIMEFRAME_M15,
        "sl_mult":  2.0,
        "tp_mult":  2.5,
        "hold":     15,
        "minutes":  15,
    },
    
    "M30": {
        "tf_const": TIMEFRAME_M30,
        "sl_mult":  2.0,
        "tp_mult":  2.5,
        "hold":     10,
        "minutes":  30,
    },
    
    "H1": {
        "tf_const": TIMEFRAME_H1,
        "sl_mult":  2.0,
        "tp_mult":  2.5,
        "hold":     10,
        "minutes":  60,
    },
    
    "H4": {
        "tf_const": TIMEFRAME_H4,
        "sl_mult":  2.0,
        "tp_mult":  3.0,
        "hold":     15,
        "minutes":  240,
    },

    "D1": {
        "tf_const": TIMEFRAME_D1,
        "sl_mult":  2.0,
        "tp_mult":  3.0,
        "hold":     10,
        "minutes":  1440,
    },

}

# ── Pattern filtering thresholds ─────────────────────────────────────────
MIN_WIN_RATE      = 0.4
MIN_PROFIT_FACTOR = 1.3
MIN_TRADES        = 10

# ── Sessions (Broker Time hours) ───────────────────────────────────────────────────────────
# Used for per-session pattern filtering: if a pattern loses money
# during a session, it gets killed for that session only.
SESSIONS = {
    "Asian":   (0, 8),    # 00:00 - 08:00 UTC
    "London":  (8, 16),   # 08:00 - 16:00 UTC
    "NewYork": (13, 21),  # 13:00 - 21:00 UTC  (overlaps London)
}
# Global time filter: no new trades outside this window
TRADE_START_HOUR = 4    # 4:00 AM Broker Time
TRADE_END_HOUR   = 20   # 8:00 PM Broker Time

# ── ATR / Indicators ────────────────────────────────────────────────────
ATR_PERIOD       = 14
RSI_PERIOD       = 14
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
EMA_FAST         = 20
EMA_SLOW         = 50
BOLLINGER_PERIOD = 20
BOLLINGER_STD    = 2.0
ADX_PERIOD       = 14
SWING_ORDER      = 5

# ── Output ──────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
CACHE_DIR   = "cache"
