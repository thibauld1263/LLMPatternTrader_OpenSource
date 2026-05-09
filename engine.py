import os
import sys
import time
import json
import asyncio
import logging
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal

import config
from data_fetcher import MT5DataFetcher
from patterns.registry import PatternRegistry
from backtester import Backtester
from services.llm_agent import LLMRiskAgent
from services.order_executor import MT5OrderExecutor
import ta_utils as utils
from news_filter import get_blocked_currencies, is_symbol_blocked, fetch_news_from_server

logger = logging.getLogger(__name__)

class TradingEngine(QObject):
    """
    Connects the PatternFinder pipeline and the PyQt5 Dashboard.
    Manages data fetching, pattern scanning, and LLM analysis loops.
    """
    status_update = pyqtSignal(str, str)
    positions_update = pyqtSignal(dict)
    connection_status = pyqtSignal(bool)
    finished = pyqtSignal()
    chart_update = pyqtSignal(str, str, object, dict, list) # symbol, tf_name, pd.DataFrame, dict of patterns, list of trade dicts
    equity_update = pyqtSignal(list, list) # dates, pnl_values
    trade_list_update = pyqtSignal(list) # list of trade dicts
    signal_update = pyqtSignal(list) # list of signal dicts for LLM Signals tab
    pattern_results_update = pyqtSignal(list) # list of pattern result dicts for Pattern Results tab
    console_ready = pyqtSignal()  # Emitted when first pattern scan is complete
    training_complete = pyqtSignal(dict)  # Emitted with validated_cache after training finishes

    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self._is_running = False
        self._validated_cache = {}  # Exposed for save from GUI
        self.registry = PatternRegistry()
        self.fetcher = MT5DataFetcher()
        self.executor = MT5OrderExecutor()

    @staticmethod
    def save_training_state(filepath, validated_cache, session_info):
        """Save validated_cache + session metadata to a JSON file."""
        # Convert tuple keys to strings for JSON serialization
        serializable_cache = {}
        for (symbol, tf_name), patterns in validated_cache.items():
            key = f"{symbol}|{tf_name}"
            serializable_cache[key] = patterns
        
        data = {
            "version": 1,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_info": session_info,
            "validated_cache": serializable_cache
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return True

    @staticmethod
    def load_training_state(filepath):
        """Load validated_cache from a JSON file. Returns (validated_cache dict, session_info dict)."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_cache = data.get("validated_cache", {})
        validated_cache = {}
        for key, patterns in raw_cache.items():
            parts = key.split("|", 1)
            if len(parts) == 2:
                validated_cache[(parts[0], parts[1])] = patterns
        
        return validated_cache, data.get("session_info", {})

    def get_open_positions_context(self, df_cache: dict, magic_number: int = 123456) -> list:
        """
        Gathers live market context for all open MT5 positions with our magic number.
        Returns list of position dicts ready for LLM monitoring.
        df_cache: dict of {symbol: DataFrame} from the current cycle's data fetches.
        """
        import MetaTrader5 as mt5
        positions = mt5.positions_get()
        if not positions:
            return []
        
        position_data = []
        for pos in positions:
            if pos.magic != magic_number:
                continue
            
            sym = pos.symbol
            tick = mt5.symbol_info_tick(sym)
            sym_info = mt5.symbol_info(sym)
            if not tick or not sym_info:
                continue
            
            current_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            floating_pnl = pos.profit
            entry_price = pos.price_open
            
            # Calculate floating PnL in pips
            point = sym_info.point
            pip_value = sym_info.point * 10
            if pos.type == mt5.ORDER_TYPE_BUY:
                pnl_pips = (current_price - entry_price) / pip_value
            else:
                pnl_pips = (entry_price - current_price) / pip_value
            
            # Get indicators if we have cached data for this symbol
            ctx = {}
            df = df_cache.get(sym)
            if df is not None and len(df) > 200:
                ema20 = utils.ema(df['close'], 20).iloc[-1]
                ema50 = utils.ema(df['close'], 50).iloc[-1]
                ema200 = utils.ema(df['close'], 200).iloc[-1]
                rsi_val = utils.rsi(df, 14).iloc[-1]
                bb_up, bb_mid, bb_low = utils.bollinger(df, 20, 2.0)
                macd_line, macd_signal, macd_hist = utils.macd(df)
                adx_val, plus_di, minus_di = utils.adx(df)
                atr_val = utils.atr(df, 14).iloc[-1]
                spread = (tick.ask - tick.bid) / sym_info.point if sym_info.point > 0 else 0
                spread_in_price = tick.ask - tick.bid
                mid = (tick.ask + tick.bid) / 2
                spread_pct = (spread_in_price / mid) * 100 if mid > 0 else 0
                
                ctx = {
                    "current_price": round(current_price, 5),
                    "trend_ema_20": round(ema20, 5),
                    "trend_ema_50": round(ema50, 5),
                    "trend_ema_200": round(ema200, 5),
                    "rsi_14": round(rsi_val, 2),
                    "bb_upper": round(bb_up.iloc[-1], 5),
                    "bb_mid": round(bb_mid.iloc[-1], 5),
                    "bb_lower": round(bb_low.iloc[-1], 5),
                    "macd_line": round(macd_line.iloc[-1], 6),
                    "macd_signal": round(macd_signal.iloc[-1], 6),
                    "macd_histogram": round(macd_hist.iloc[-1], 6),
                    "adx_14": round(adx_val.iloc[-1], 2),
                    "plus_di": round(plus_di.iloc[-1], 2),
                    "minus_di": round(minus_di.iloc[-1], 2),
                    "atr_14": round(atr_val, 5),
                    "spread_points": round(spread, 1),
                    "spread_percentage": round(spread_pct, 4)
                }
            
            position_data.append({
                "ticket": pos.ticket,
                "symbol": sym,
                "direction": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                "entry_price": round(entry_price, 5),
                "current_price": round(current_price, 5),
                "sl": round(pos.sl, 5),
                "tp": round(pos.tp, 5),
                "volume": pos.volume,
                "floating_pnl_pips": round(pnl_pips, 1),
                "floating_pnl_usd": round(floating_pnl, 2),
                "comment": pos.comment,
                "market_context": ctx
            })
        
        return position_data

    def calculate_atr_risk_metrics(self, df: pd.DataFrame, symbol: str, current_price: float, is_long: bool) -> dict:
        """
        Calculates strict Risk options for the LLM based on Average True Range.
        Generates 1.5RR and 2.0RR bound options.
        """
        atr_series = utils.atr(df, period=14)
        current_atr = atr_series.iloc[-1]
        
        # Standard Risk Profile: SL = 2x ATR, TP = 3x ATR (1.5 RR)
        # Max Risk Profile: SL = 2x ATR, TP = 4x ATR (2.0 RR)
        sl_dist = current_atr * 2.0
        
        # Enforce minimum 10-pip SL distance
        pip_size, _ = MT5DataFetcher.pip_info(symbol)
        min_sl_dist = pip_size * 10
        if sl_dist < min_sl_dist:
            sl_dist = min_sl_dist
        
        if is_long:
            sl_price = current_price - sl_dist
            tp_standard = current_price + (current_atr * 2.0 * 1.5)
            tp_max = current_price + (current_atr * 2.0 * 2.0)
        else:
            sl_price = current_price + sl_dist
            tp_standard = current_price - (current_atr * 2.0 * 1.5)
            tp_max = current_price - (current_atr * 2.0 * 2.0)
            
        import MetaTrader5 as mt5
        account = mt5.account_info()
        balance = account.balance if account else 0.0
        
        symbol_info = mt5.symbol_info(symbol)
        min_lot = symbol_info.volume_min if symbol_info else 0.01
        lot_step = symbol_info.volume_step if symbol_info else 0.01
        digits = symbol_info.digits if symbol_info else 5
        
        # Calculate SL/TP distance in pips for display
        sl_pips = round(sl_dist / pip_size, 1)
        tp_std_pips = round(abs(tp_standard - current_price) / pip_size, 1)
        tp_max_pips = round(abs(tp_max - current_price) / pip_size, 1)
        
        return {
            "atr_value": round(current_atr, digits),
            "atr_pips": round(current_atr / pip_size, 1),
            "stop_loss": round(sl_price, digits),
            "sl_pips": sl_pips,
            "tp_1_5_rr": round(tp_standard, digits),
            "tp_1_5_pips": tp_std_pips,
            "tp_2_0_rr": round(tp_max, digits),
            "tp_2_0_pips": tp_max_pips,
            "account_balance": round(balance, 2),
            "min_lot_size": min_lot,
            "lot_step": lot_step,
            "digits": digits,
            "pip_size": pip_size
        }
    
    # Higher TF hierarchy for context injection
    HTF_HIERARCHY = {
        "M5":  ["H1", "H4"],
        "M15": ["H1", "H4"],
        "M30": ["H1", "H4"],
        "H1":  ["H4", "D1"],
        "H4":  ["D1"],
    }
    
    def get_htf_context(self, symbol: str, signal_tf: str) -> list:
        """Fetch higher timeframe indicator context for a symbol.
        Returns list of dicts with key indicators per HTF."""
        htf_list = self.HTF_HIERARCHY.get(signal_tf, [])
        if not htf_list:
            return []
        
        htf_contexts = []
        for htf_name in htf_list:
            tf_cfg = config.TIMEFRAMES.get(htf_name)
            if not tf_cfg:
                continue
            try:
                df = self.fetcher.fetch_symbol(
                    symbol, tf_cfg["tf_const"], 300, htf_name, use_cache=True
                )
                if df is None or len(df) < 200:
                    continue
                
                ema20 = utils.ema(df['close'], 20).iloc[-1]
                ema50 = utils.ema(df['close'], 50).iloc[-1]
                ema200 = utils.ema(df['close'], 200).iloc[-1]
                rsi_val = utils.rsi(df, 14).iloc[-1]
                adx_val, plus_di, minus_di = utils.adx(df)
                atr_val = utils.atr(df, 14).iloc[-1]
                
                # Determine trend from EMA structure
                price = df['close'].iloc[-1]
                if price > ema20 > ema50 > ema200:
                    trend = "Bullish (EMAs stacked up)"
                elif price < ema20 < ema50 < ema200:
                    trend = "Bearish (EMAs stacked down)"
                elif price > ema200:
                    trend = "Above EMA200 (broadly bullish)"
                elif price < ema200:
                    trend = "Below EMA200 (broadly bearish)"
                else:
                    trend = "Neutral/Ranging"
                
                htf_contexts.append({
                    "timeframe": htf_name,
                    "trend": trend,
                    "ema_20": round(ema20, 5),
                    "ema_50": round(ema50, 5),
                    "ema_200": round(ema200, 5),
                    "rsi_14": round(rsi_val, 2),
                    "adx_14": round(adx_val.iloc[-1], 2),
                    "plus_di": round(plus_di.iloc[-1], 2),
                    "minus_di": round(minus_di.iloc[-1], 2),
                    "atr_14": round(atr_val, 5)
                })
            except Exception as e:
                logger.debug(f"HTF context fetch failed for {symbol} {htf_name}: {e}")
                continue
        
        return htf_contexts    
    # Risk mode definitions: {mode_name: (min_risk%, max_risk%)}
    # The LLM outputs a grade (A to AAA+). Python maps grade → numeric (1-6)
    # then linearly maps to the mode's (min, max) to get ACTUAL risk %.
    # Max 2.5% per trade no matter what.
    RISK_MODES = {
        "Fix Lot Size": None,
        "Ultra Safe":    (0.04, 0.25),
        "Conservative":  (0.08, 0.50),
        "Moderate":      (0.15, 1.00),
        "Aggressive":    (0.25, 1.50),
        "Max Risk":      (0.50, 2.50),
    }
    
    PORTFOLIO_CAPS = {
        "Fix Lot Size": 25.0,
        "Ultra Safe":    2.0,
        "Conservative":  5.0,
        "Moderate":      8.0,
        "Aggressive":   12.0,
        "Max Risk":     15.0,
    }
    
    # Grade → numeric value (1-6)
    GRADE_MAP = {
        "A": 1, "A+": 2, "AA": 3, "AA+": 4, "AAA": 5, "AAA+": 6
    }

    def map_llm_risk_to_actual(self, grade_value: float, risk_mode: str) -> float:
        """Map LLM grade (1-6) to actual risk % based on risk mode.
        
        Linear interpolation: grade 1 → mode min, grade 6 → mode max.
        Example: grade=4 (AA+), mode="Moderate" (0.15, 1.00) → 0.66%
        """
        mode_range = self.RISK_MODES.get(risk_mode)
        if mode_range is None:  # Fix Lot Size
            return grade_value
        min_pct, max_pct = mode_range
        clamped = max(1, min(int(grade_value), 6))
        t = (clamped - 1) / 5.0  # 0.0 to 1.0
        return min_pct + t * (max_pct - min_pct)

    def calculate_lots_for_risk(self, symbol: str, balance: float, stop_loss: float,
                                is_long: bool, actual_risk_pct: float,
                                fixed_lot: float = 0.01, risk_mode: str = "Moderate") -> float:
        """Calculate exact lot size for a specific risk percentage.
        
        Args:
            actual_risk_pct: The ACTUAL risk %, already mapped through the risk mode.
            Returns: The lot size (float), properly rounded to lot_step.
        """
        import MetaTrader5 as mt5
        
        if risk_mode == "Fix Lot Size":
            return fixed_lot
        
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return 0.01
        
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        min_lot = symbol_info.volume_min
        lot_step = symbol_info.volume_step
        max_lot = symbol_info.volume_max
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return min_lot
        price = tick.ask if is_long else tick.bid
        
        sl_distance = abs(price - float(stop_loss))
        if sl_distance <= 0 or tick_size <= 0 or tick_value <= 0:
            return min_lot
        
        sl_ticks = sl_distance / tick_size
        risk_cash = balance * (actual_risk_pct / 100.0)
        calc_vol = risk_cash / (sl_ticks * tick_value)
        
        final_volume = round(calc_vol / lot_step) * lot_step
        final_volume = max(min_lot, min(final_volume, max_lot))
        return round(final_volume, 2)

    def calculate_tier_lots(self, symbol: str, balance: float, stop_loss: float, is_long: bool, risk_mode: str = "Extreme", fixed_lot: float = 0.01) -> dict:
        """Pre-calculate lot sizes for the 3 display tiers (1%, 2%, 3% LLM conviction)
        mapped through the current risk mode. Used for previewing in the LLM prompt."""
        if risk_mode == "Fix Lot Size":
            return {"1%": fixed_lot, "2%": fixed_lot, "3%": fixed_lot}
        
        lots = {}
        for llm_pct, key in [(1.0, "1%"), (2.0, "2%"), (3.0, "3%")]:
            actual = self.map_llm_risk_to_actual(llm_pct, risk_mode)
            lots[key] = self.calculate_lots_for_risk(symbol, balance, stop_loss, is_long, actual, fixed_lot, risk_mode)
        return lots

    def session_filter(self, signals: pd.DataFrame, bt: Backtester,
                       df: pd.DataFrame, sym: str,
                       passed_global: list) -> tuple:
        """For each surviving pattern, test performance per session.
        If a pattern loses money (PF < 1.0) during a session,
        zero out its signals during that session's hours."""

        filtered = signals.copy()
        hours = df.index.hour
        session_kills = []

        for pat_name in passed_global:
            for sess_name, (h_start, h_end) in config.SESSIONS.items():
                # Get signals only during this session
                if h_start < h_end:
                    in_session = (hours >= h_start) & (hours < h_end)
                else:  # wraps midnight
                    in_session = (hours >= h_start) | (hours < h_end)

                sess_sig = pd.Series(0, index=df.index)
                sess_sig[in_session] = signals[pat_name][in_session]

                if (sess_sig != 0).sum() < 5:
                    continue  # not enough signals to judge

                result = bt.run(df, sess_sig, sym, pat_name)

                if result.trade_count >= 5 and result.profit_factor < 1.0:
                    # Kill this pattern during this session
                    filtered.loc[in_session, pat_name] = 0
                    session_kills.append(f"{pat_name}@{sess_name}")

        return filtered, session_kills

    async def start_with_loop(self, session_config, loop):
        """Called when 'RUN' is clicked."""
        self._is_running = True
        self.status_update.emit("Connecting to MT5...", "status_label_pending")
        
        # Connect to MT5
        if not self.fetcher.initialize(path=session_config.get("mt4_data_path", "")):
            logger.error("Failed to connect to MT5.")
            self.status_update.emit("MT5 Connection Failed", "status_label_error")
            self.connection_status.emit(False)
            self._finish()
            return
            
        self.connection_status.emit(True)
        self.status_update.emit("Scanning Patterns...", "status_label_pending")
        
        try:
            # Only loop on the configured symbols and timeframes
            symbols = session_config.get("trading_symbols", config.SYMBOLS)
            tfs_to_run = session_config.get("trading_timeframes", ["H1"])
            
            # Trading hours from UI (override config defaults)
            trade_start_hour = session_config.get("trade_start_hour", config.TRADE_START_HOUR)
            trade_end_hour = session_config.get("trade_end_hour", config.TRADE_END_HOUR)
            
            # Risk mode from UI
            risk_mode = session_config.get("risk_mode", "Extreme")
            fixed_lot = session_config.get("fixed_lot", 0.01)
            signal_only = session_config.get("signal_only", False)
            llm_filter = session_config.get("llm_filter", True)
            magic_number = session_config.get("magic_number", 123456)
            trade_comment = session_config.get("trade_comment", "PT")
            logger.info({"type": "observation", "message": f"Risk Mode: {risk_mode}" + (f" (fixed {fixed_lot} lots)" if risk_mode == 'Fix Lot Size' else f" ({self.RISK_MODES.get(risk_mode, (1,2,3))}%)") + (" | SIGNAL ONLY MODE" if signal_only else "") + (" | LLM FILTER ON" if llm_filter else " | LLM FILTER OFF (direct trading)")})
            
            first_run = True
            all_session_trades = []
            
            # News buffer settings from session config
            news_before = session_config.get("news_buffer_before", 120)
            news_after = session_config.get("news_buffer_after", 120)
            
            # Fetch news from server
            self.status_update.emit("Fetching news calendar from server...", "status_label_pending")
            logger.info({"type": "observation", "message": "Fetching news calendar from server..."})
            news_ok = fetch_news_from_server()
            if not news_ok:
                logger.error("CRITICAL: Could not fetch news calendar. Trading cannot proceed without news data.")
                self.status_update.emit("ERROR: News calendar unavailable. Check internet connection.", "status_label_error")
                return
            logger.info({"type": "observation", "message": f"News loaded. Buffer: {news_before}min before / {news_after}min after."})
            
            # Cache for validated patterns: {(symbol, tf_name): {pat_name: {win_rate, pf, trades}}}
            validated_cache = {}
            # Cache for trade list data (for the UI tab)  
            trade_list_data = []

            # Pre-populate validated_cache from loaded training state (if any)
            loaded_cache = session_config.get("loaded_training_cache")
            if loaded_cache:
                validated_cache.update(loaded_cache)
                logger.info({"type": "observation", "message": f"Loaded {len(loaded_cache)} pre-trained symbol/TF pairs from saved state."})

            # Record when training starts so we can catch up on missed candles
            # Only if there are actually NEW symbols to train (not all loaded from file)
            from news_filter import get_broker_time as _get_bt
            import datetime as _dt_module
            
            # Check if every symbol/TF combo is already in the loaded cache
            all_loaded = loaded_cache and all(
                (sym, tf) in validated_cache
                for tf in tfs_to_run if tf in config.TIMEFRAMES
                for sym in symbols
            )
            
            if all_loaded:
                training_started_at = None
                logger.info({"type": "observation", "message": "All symbols loaded from saved state — skipping catch-up phase."})
            else:
                training_started_at = _get_bt()
                logger.info({"type": "observation", "message": f"Training start recorded at broker time: {training_started_at.strftime('%H:%M:%S')}"})

            while self._is_running:
                import datetime
                from news_filter import get_broker_time
                
                # WEEKEND GUARD: Only block LIVE TRADING (subsequent runs).
                # Training/backtesting (first_run) always proceeds — it uses historical data.
                if not first_run:
                    utc_now = datetime.datetime.utcnow()
                    if utc_now.isoweekday() in (6, 7):
                        self.status_update.emit("Weekend — markets closed. Sleeping...", "status_label_pending")
                        logger.info({"type": "observation", "message": "Weekend detected — sleeping 5 minutes before re-checking."})
                        slept = 0
                        while slept < 300 and self._is_running:
                            await asyncio.sleep(1)
                            slept += 1
                        continue
                
                now = get_broker_time()
                
                valid_tfs = {k: v for k, v in config.TIMEFRAMES.items() if k in tfs_to_run}
            
                min_minutes = 60
                active_found = False
                for tf_name in valid_tfs:
                    val = config.TIMEFRAMES[tf_name].get("minutes", 60)
                    if not active_found or val < min_minutes:
                        min_minutes = val
                        active_found = True

                # Wait for next candle close (skip only on the very first iteration)
                if not first_run:
                    from news_filter import get_broker_time
                    now = get_broker_time()
                    total_minutes = now.hour * 60 + now.minute
                    next_boundary_total = ((total_minutes // min_minutes) + 1) * min_minutes
                    minutes_to_add = next_boundary_total - total_minutes
                    next_target = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=minutes_to_add)
                    sleep_seconds = (next_target - now).total_seconds()
                    
                    # Add 5 seconds buffer to ensure the candle has fully closed
                    sleep_seconds = max(0, sleep_seconds) + 5
                    
                    self.status_update.emit(f"Waiting for next M{min_minutes} candle close...", "status_label_pending")
                    slept_so_far = 0
                    while slept_so_far < sleep_seconds and self._is_running:
                        await asyncio.sleep(1)
                        slept_so_far += 1
                            
                    if not self._is_running:
                        break
                else:
                    # First run starts IMMEDIATELY — no waiting for candle
                    self.status_update.emit("Running initial backtest & pattern discovery...", "status_label_ok")
                    logger.info({"type": "observation", "message": "First run — starting immediately (no candle wait)."})
                
                # Cache DataFrames for position monitoring context
                df_cache = {}
                # Gather open positions for LLM monitoring
                open_positions = []
                

                
                # Log news blackout status at the start of each cycle
                from news_filter import get_blocked_currencies, get_broker_time
                broker_now = get_broker_time()
                blocked_now = get_blocked_currencies(broker_now, buffer_before=news_before, buffer_after=news_after)
                if blocked_now:
                    logger.warning(f"NEWS BLACKOUT ACTIVE (Broker Time {broker_now.strftime('%H:%M')}): {', '.join(sorted(blocked_now))} — {'ALL PAIRS BLOCKED (entries only)' if 'USD' in blocked_now else 'affected pairs blocked (entries only)'}")
                
                for tf_name, tf_cfg in valid_tfs.items():
                    if not self._is_running:
                        break
                        
                    current_batch = []
                        
                    logger.info({"type": "observation", "message": f"Fetching data for TF: {tf_name}..."})
                
                    for symbol in symbols:
                        if not self._is_running:
                            break
                        
                        cache_key = (symbol, tf_name)
                        
                        if first_run:
                            # Loaded symbol: skip TRAINING but still fetch data + display backtest
                            if cache_key in validated_cache:
                                loaded_patterns = list(validated_cache[cache_key].keys())
                                logger.info({"type": "observation", "message": f"Loaded {symbol} ({tf_name}) from saved state ({len(loaded_patterns)} patterns). Running display backtest..."})
                                self.status_update.emit(f"Loading {symbol} ({tf_name})...", "status_label_pending")
                                
                                # Fetch fresh data for chart display
                                bars_to_fetch = session_config.get("bars_to_fetch", 20000)
                                df = self.fetcher.fetch_data(symbol, tf_name, use_cache=False, bars=bars_to_fetch)
                                if df is None or df.empty:
                                    logger.warning(f"No data for loaded symbol {symbol} on {tf_name}.")
                                    continue
                                df_cache[symbol] = df
                                
                                # Scan only the loaded patterns for display
                                signals = self.registry.scan_specific(df, loaded_patterns)
                                
                                # Time filter
                                hours = df.index.hour
                                outside = (hours < trade_start_hour) | (hours >= trade_end_hour)
                                signals.loc[outside] = 0
                                
                                # Run backtest for display
                                bt = Backtester(sl_mult=tf_cfg["sl_mult"], tp_mult=tf_cfg["tp_mult"], max_hold=tf_cfg["hold"] * 5)
                                results = bt.run_all_patterns(df, signals, symbol)
                                
                                # Emit chart — emit after combined backtest so trades overlay is available
                                plot_data = {pat: signals[pat] for pat in loaded_patterns if pat in signals.columns}
                                
                                # Emit pattern results
                                combined_result = None
                                if results:
                                    stats_df = bt.results_table(results)
                                    pattern_rows = []
                                    for _, row in stats_df.iterrows():
                                        pattern_rows.append({
                                            "symbol": symbol,
                                            "pattern": row["Pattern"],
                                            "trades": int(row["Trades"]),
                                            "win_rate": row["Win Rate"],
                                            "profit_factor": row["Profit Factor"],
                                            "cumul_balance": row["Cumulative Balance"],
                                            "avg_gain": row["Avg Gain"],
                                            "max_dd": row["Max DD"]
                                        })
                                    self.pattern_results_update.emit(pattern_rows)
                                    
                                    # Combined backtest for equity curve
                                    pattern_scores = {p: r.profit_factor for p, r in results.items()}
                                    combined_result = bt.run_combined(df, signals[[p for p in results.keys() if p in signals.columns]], symbol, pattern_scores)
                                
                                # Build trade dicts for chart overlay
                                chart_trades = []
                                if combined_result and combined_result.trades:
                                    for t in combined_result.trades:
                                        chart_trades.append({
                                            "entry_time": t.entry_time,
                                            "exit_time": t.exit_time,
                                            "entry_price": t.entry_price,
                                            "exit_price": t.exit_price,
                                            "direction": t.direction,
                                            "outcome": t.outcome,
                                            "pnl_pips": t.pnl_pips,
                                            "pattern": t.pattern
                                        })
                                self.chart_update.emit(symbol, tf_name, df, plot_data, chart_trades)
                                
                                if combined_result:
                                    all_session_trades.extend(combined_result.trades)
                                    
                                    for t in combined_result.trades:
                                        trade_list_data.append({
                                            "date": t.entry_time.strftime('%Y-%m-%d %H:%M') if hasattr(t, 'entry_time') and t.entry_time else "",
                                            "symbol": symbol,
                                            "pattern": t.pattern,
                                            "direction": t.direction,
                                            "entry": t.entry_price,
                                            "exit": t.exit_price if t.exit_price else 0,
                                            "pnl": t.pnl_pips,
                                            "outcome": t.outcome
                                        })
                                    
                                    # Emit equity + trade list
                                    if all_session_trades:
                                        sorted_trades = sorted([t for t in all_session_trades if t.exit_time is not None], key=lambda t: t.exit_time)
                                        if sorted_trades:
                                            import numpy as np
                                            exits_idx = list(range(len(sorted_trades)))
                                            pnls = [t.pnl_pips for t in sorted_trades]
                                            cum = np.cumsum(pnls).tolist()
                                            self.equity_update.emit(exits_idx, cum)
                                    if trade_list_data:
                                        self.trade_list_update.emit(trade_list_data)
                                
                                continue
                            
                            # Full data fetch for backtesting (first run only)
                            logger.info(f"Downloading {symbol} ({tf_name})")
                            bars_to_fetch = session_config.get("bars_to_fetch", 20000)
                            df = self.fetcher.fetch_data(symbol, tf_name, use_cache=False, bars=bars_to_fetch)
                            if df is None or df.empty:
                                logger.warning(f"No data for {symbol} on {tf_name}.")
                                continue
                            df_cache[symbol] = df
                            # ============ FIRST RUN: Full Backtest + Validation ============
                            self.status_update.emit(f"Scanning {symbol} ({tf_name})...", "status_label_pending")
                            logger.info(f"Scanning {symbol} for patterns...")
                            signals = self.registry.scan(df)
                            
                            # 1. Global Time Filter
                            hours = df.index.hour
                            outside = (hours < trade_start_hour) | (hours >= trade_end_hour)
                            signals.loc[outside] = 0
                            
                            n_signals = (signals != 0).sum().sum()
                            logger.info({"type": "observation", "message": f"Found {n_signals} signals on {symbol} (Time Filtered)."})
                            
                            # 2. Backtest & Threshold Filter
                            bt = Backtester(sl_mult=tf_cfg["sl_mult"], tp_mult=tf_cfg["tp_mult"], max_hold=tf_cfg["hold"] * 5)
                            results = bt.run_all_patterns(df, signals, symbol)
                            
                            if results:
                                stats_df = bt.results_table(results)
                                
                                filtered_df = stats_df[
                                    (stats_df["Win Rate"] >= session_config["min_win_rate"]) &
                                    (stats_df["Profit Factor"] >= session_config["min_profit_factor"]) &
                                    (stats_df["Trades"] >= session_config["min_trades"])
                                ]
                                
                                if not filtered_df.empty:
                                    print(f"\n=== {symbol} {tf_name} VALIDATED PATTERNS ===")
                                    with pd.option_context('display.max_columns', None, 'display.width', 1000):
                                        print(filtered_df.to_string(index=False))
                                    print("===================================================\n")
                                    
                                    # Emit pattern results to UI
                                    pattern_rows = []
                                    for _, row in filtered_df.iterrows():
                                        pattern_rows.append({
                                            "symbol": symbol,
                                            "pattern": row["Pattern"],
                                            "trades": int(row["Trades"]),
                                            "win_rate": row["Win Rate"],
                                            "profit_factor": row["Profit Factor"],
                                            "cumul_balance": row["Cumulative Balance"],
                                            "avg_gain": row["Avg Gain"],
                                            "max_dd": row["Max DD"]
                                        })
                                    self.pattern_results_update.emit(pattern_rows)
                            
                            passed = []
                            for pat, r in results.items():
                                if (r.win_rate >= session_config["min_win_rate"] and
                                    r.profit_factor >= session_config["min_profit_factor"] and
                                    r.trade_count >= session_config["min_trades"]):
                                    passed.append(pat)

                            if not passed:
                                logger.info(f"No patterns survived the thresholds for {symbol}.")
                                continue
                                
                            logger.info({"type": "observation", "message": f"{len(passed)} patterns survived global thresholds. Running Session Filters..."})
                            
                            # 3. Session Filters
                            filtered_signals, session_kills = self.session_filter(signals, bt, df, symbol, passed)
                            if session_kills:
                                logger.info({"type": "observation", "message": f"Session kills: {', '.join(session_kills)}"})

                            # Emit validated pattern chart to the UI
                            plot_data = {pat: filtered_signals[pat] for pat in passed}

                            # 4. Final Re-backtest — individual runs for per-pattern stats
                            survived = {}
                            for pat in passed:
                                pat_sig = filtered_signals[pat]
                                if (pat_sig != 0).sum() < session_config["min_trades"]:
                                    continue
                                    
                                r = bt.run(df, pat_sig, symbol, pat)
                                if (r.win_rate >= session_config["min_win_rate"] and
                                    r.profit_factor >= session_config["min_profit_factor"] and
                                    r.trade_count >= session_config["min_trades"]):
                                    survived[pat] = r
                            
                            # 5. Combined backtest: 1 trade per bar per symbol
                            #    Prioritize by profit factor when multiple patterns fire same bar
                            chart_trades = []
                            if survived:
                                survived_signals = filtered_signals[[p for p in survived.keys() if p in filtered_signals.columns]]
                                pattern_scores = {p: r.profit_factor for p, r in survived.items()}
                                combined_result = bt.run_combined(df, survived_signals, symbol, pattern_scores)
                                all_session_trades.extend(combined_result.trades)
                                
                                # Build trade dicts for chart overlay
                                for t in combined_result.trades:
                                    chart_trades.append({
                                        "entry_time": t.entry_time,
                                        "exit_time": t.exit_time,
                                        "entry_price": t.entry_price,
                                        "exit_price": t.exit_price,
                                        "direction": t.direction,
                                        "outcome": t.outcome,
                                        "pnl_pips": t.pnl_pips,
                                        "pattern": t.pattern
                                    })
                                
                                # Collect trade data for the Trade List tab
                                for t in combined_result.trades:
                                    trade_list_data.append({
                                        "date": t.entry_time.strftime('%Y-%m-%d %H:%M') if hasattr(t, 'entry_time') and t.entry_time else "",
                                        "symbol": symbol,
                                        "pattern": t.pattern,
                                        "direction": t.direction,
                                        "entry": t.entry_price,
                                        "exit": t.exit_price if t.exit_price else 0,
                                        "pnl": t.pnl_pips,
                                        "outcome": t.outcome
                                    })

                            # Emit chart with trade overlay
                            self.chart_update.emit(symbol, tf_name, df, plot_data, chart_trades)

                            # Emit cumulative equity curve progressively
                            if all_session_trades:
                                sorted_trades = sorted([t for t in all_session_trades if t.exit_time is not None], key=lambda t: t.exit_time)
                                if sorted_trades:
                                    import numpy as np
                                    exits_idx = list(range(len(sorted_trades)))
                                    pnls = [t.pnl_pips for t in sorted_trades]
                                    cum = np.cumsum(pnls).tolist()
                                    self.equity_update.emit(exits_idx, cum)
                            
                            # Emit trade list to UI
                            if trade_list_data:
                                self.trade_list_update.emit(trade_list_data)

                            if not survived:
                                logger.info(f"No patterns survived the session filter for {symbol}.")
                                continue
                            
            # CACHE validated patterns WITH their backtest stats
                            validated_cache[cache_key] = {
                                pat_name: {
                                    "win_rate": r.win_rate,
                                    "profit_factor": r.profit_factor,
                                    "trade_count": r.trade_count,
                                    "avg_gain": r.avg_gain
                                } for pat_name, r in survived.items()
                            }

                            logger.info({"type": "observation", "message": f"{len(survived)} patterns formally survived all metrics for {symbol}."})
                            
                        else:
                            # ============ SUBSEQUENT RUNS: Lightweight Signal Check ============
                            cached_data = validated_cache.get(cache_key, {})
                            if not cached_data:
                                continue
                            cached_patterns = list(cached_data.keys())
                            
                            # NEWS BLACKOUT flag: skip NEW ENTRIES but still fetch data and update charts
                            news_blocked = is_symbol_blocked(symbol, buffer_before=news_before, buffer_after=news_after)
                            if news_blocked:
                                logger.info({"type": "observation", "message": f"🚫 {symbol} blocked — news blackout (entries only). Data still updating."})
                            
                            # Fetch minimal data: 500 bars is enough for EMA 200 + indicators
                            self.status_update.emit(f"Checking {symbol} ({tf_name})...", "status_label_pending")
                            logger.info(f"Checking {len(cached_patterns)} cached patterns for {symbol}...")
                            
                            tf_cfg_local = config.TIMEFRAMES.get(tf_name, {})
                            df = self.fetcher.fetch_symbol(
                                symbol, tf_cfg_local.get("tf_const", 16385), 
                                500, tf_name, use_cache=False
                            )
                            if df is None or df.empty:
                                continue
                            
                            # Update cache for position monitoring
                            df_cache[symbol] = df
                            
                            # Scan ONLY the validated patterns — not all 78+
                            signals = self.registry.scan_specific(df, cached_patterns)
                            
                            # Time filter
                            hours = df.index.hour
                            outside = (hours < trade_start_hour) | (hours >= trade_end_hour)
                            signals.loc[outside] = 0
                            
                            # Use cached backtest stats (real numbers, not None)
                            survived = cached_data
                            
                            # Update chart with fresh live candles + current active signals
                            live_plot_data = {p: signals[p] for p in cached_patterns if p in signals.columns}
                            self.chart_update.emit(symbol, tf_name, df, live_plot_data, [])
                            
                            # If news blocked, skip signal processing but keep data fresh
                            if news_blocked:
                                continue
                        
                        # ============ COMMON: Check for Active Signal on Current Candle ============
                        # NEVER fire signals during the backtest phase — only in live mode.
                        if first_run:
                            continue
                        
                        active_signals_found = 0
                        active_patterns_map = {}
                        for pat in list(survived.keys()):
                            
                            # Check if the pattern column exists in signals
                            if pat not in signals.columns:
                                continue
                            
                            # CRITICAL: Only evaluate if the pattern has an ACTIVE signal on the current live candle.
                            last_sig = signals[pat].iloc[-1]
                            if last_sig == 0:
                                continue
                                
                            active_signals_found += 1
                            active_patterns_map[pat] = signals[pat]
                            
                            # Gather rich contextual market intelligence for the LLM
                            ema20 = utils.ema(df['close'], 20).iloc[-1]
                            ema50 = utils.ema(df['close'], 50).iloc[-1]
                            ema200 = utils.ema(df['close'], 200).iloc[-1]
                            rsi_val = utils.rsi(df, 14).iloc[-1]
                            bb_up, bb_mid, bb_low = utils.bollinger(df, 20, 2.0)
                            macd_line, macd_signal, macd_hist = utils.macd(df)
                            adx_val, plus_di, minus_di = utils.adx(df)
                            atr_val = utils.atr(df, 14).iloc[-1]
                            
                            import MetaTrader5 as mt5
                            tick = mt5.symbol_info_tick(symbol)
                            sym_info = mt5.symbol_info(symbol)
                            spread = (tick.ask - tick.bid) / sym_info.point if tick and sym_info else 0.0
                            spread_in_price = tick.ask - tick.bid if tick else 0.0
                            mid_price = (tick.ask + tick.bid) / 2 if tick else df['close'].iloc[-1]
                            spread_pct = (spread_in_price / mid_price) * 100 if mid_price > 0 else 0.0
                            
                            # Use MT5 digits for proper rounding per instrument
                            sym_digits = sym_info.digits if sym_info else 5
                            
                            market_context = {
                                "current_price": round(df['close'].iloc[-1], sym_digits),
                                "trend_ema_20": round(ema20, sym_digits),
                                "trend_ema_50": round(ema50, sym_digits),
                                "trend_ema_200": round(ema200, sym_digits),
                                "rsi_14": round(rsi_val, 2),
                                "bb_upper": round(bb_up.iloc[-1], sym_digits),
                                "bb_mid": round(bb_mid.iloc[-1], sym_digits),
                                "bb_lower": round(bb_low.iloc[-1], sym_digits),
                                "macd_line": round(macd_line.iloc[-1], sym_digits + 1),
                                "macd_signal": round(macd_signal.iloc[-1], sym_digits + 1),
                                "macd_histogram": round(macd_hist.iloc[-1], sym_digits + 1),
                                "adx_14": round(adx_val.iloc[-1], 2),
                                "plus_di": round(plus_di.iloc[-1], 2),
                                "minus_di": round(minus_di.iloc[-1], 2),
                                "atr_14": round(atr_val, sym_digits),
                                "spread_points": round(spread, 1),
                                "spread_percentage": round(spread_pct, 4)
                            }
                            
                            # S/R Zones & VWAP for structural context
                            try:
                                sr = utils.sr_zones(df, lookback=100, num_zones=3)
                                vwap_val = utils.vwap(df)
                                market_context['resistance_1'] = sr.get('resistance_1')
                                market_context['resistance_2'] = sr.get('resistance_2')
                                market_context['resistance_3'] = sr.get('resistance_3')
                                market_context['support_1'] = sr.get('support_1')
                                market_context['support_2'] = sr.get('support_2')
                                market_context['support_3'] = sr.get('support_3')
                                market_context['vwap'] = vwap_val
                            except Exception:
                                pass  # Non-critical, continue without S/R zones
                            
                            # Volume context
                            try:
                                cur_vol = df['volume'].iloc[-1]
                                avg_vol = df['volume'].rolling(20).mean().iloc[-1]
                                vol_ratio = round(cur_vol / avg_vol, 2) if avg_vol and avg_vol > 0 else None
                                market_context['volume'] = int(cur_vol)
                                market_context['avg_volume_20'] = int(avg_vol) if pd.notna(avg_vol) else None
                                market_context['volume_ratio'] = vol_ratio
                            except Exception:
                                pass
                            
                            # Determine baseline risk metrics (to be verified by the AI)
                            is_long = (last_sig > 0)
                            current_price = df['close'].iloc[-1]
                            risk_params = self.calculate_atr_risk_metrics(df, symbol, current_price, is_long)
                            
                            # Signal-only: force a virtual balance so the LLM never rejects on $0
                            if signal_only and risk_params.get('account_balance', 0) == 0:
                                risk_params['account_balance'] = 10000.0
                            
                            # Pre-calculate exact Lot Sizes based on balance and SL distance before LLM acts
                            tier_lots = self.calculate_tier_lots(symbol, risk_params['account_balance'], risk_params['stop_loss'], is_long, risk_mode, fixed_lot)
                            risk_params['tier_1_lots'] = tier_lots.get("1%")
                            risk_params['tier_2_lots'] = tier_lots.get("2%")
                            risk_params['tier_3_lots'] = tier_lots.get("3%")
                            
                            r_stats = survived.get(pat)
                            # Handle both BacktestResult objects (first run) and plain dicts (subsequent)
                            if r_stats is None:
                                stats = {"win_rate": 0, "profit_factor": 0, "trade_count": 0, "avg_gain": 0}
                            elif isinstance(r_stats, dict):
                                stats = r_stats  # Already a dict from cache
                            else:
                                stats = {
                                    "win_rate": r_stats.win_rate,
                                    "profit_factor": r_stats.profit_factor,
                                    "trade_count": r_stats.trade_count,
                                    "avg_gain": r_stats.avg_gain
                                }

                            # Fetch higher TF context for LLM analysis
                            htf_ctx = self.get_htf_context(symbol, tf_name)
                            
                            current_batch.append({
                                "symbol": symbol,
                                "timeframe": tf_name,
                                "pattern": pat,
                                "signal_direction": int(last_sig),
                                "stats": stats,
                                "risk_params": risk_params,
                                "market_context": market_context,
                                "htf_context": htf_ctx
                            })

                    
                    # --- 4.5 Position Monitoring: Gather Open Positions ---
                    open_positions = self.get_open_positions_context(df_cache, magic_number)
                    if open_positions:
                        logger.info({"type": "observation", "message": f"Monitoring {len(open_positions)} open position(s)..."})
                    
                    # --- 5. Signal Evaluation (LLM or Direct) ---
                    if (current_batch or open_positions) and self._is_running:
                        
                        # ══════════════════════════════════════════════════════
                        # PATH A: LLM Filter OFF → Direct execution of all signals
                        # ══════════════════════════════════════════════════════
                        if not llm_filter and current_batch:
                            logger.info({"type": "observation", "message": f"LLM Filter OFF — Executing {len(current_batch)} signal(s) directly with ATR risk."})
                            
                            executed_symbols = set()
                            for setup in current_batch:
                                sym = setup['symbol']
                                pat = setup['pattern']
                                sig_dir = setup['signal_direction']
                                is_long = (sig_dir > 0)
                                risk_params = setup['risk_params']
                                stats = setup['stats']
                                
                                # 1 trade per pair per bar
                                if sym in executed_symbols:
                                    logger.info({"type": "observation", "message": f"SKIPPED {pat} on {sym}: already traded this pair this bar."})
                                    continue
                                
                                action = "buy" if is_long else "sell"
                                atr = risk_params['atr_value']
                                balance = risk_params['account_balance']
                                
                                import MetaTrader5 as mt5
                                tick = mt5.symbol_info_tick(sym)
                                if not tick:
                                    logger.error(f"Cannot execute {pat} on {sym}: No tick data.")
                                    continue
                                current_price = tick.ask if is_long else tick.bid
                                
                                # Default ATR multipliers (same as backtester)
                                sl_mult_exec = 2.0
                                tp_mult_exec = 3.0  # 1.5 RR
                                
                                sl_dist = atr * sl_mult_exec
                                pip_size_live, _ = MT5DataFetcher.pip_info(sym)
                                if sl_dist < pip_size_live * 10:
                                    sl_dist = pip_size_live * 10
                                
                                sym_info_exec = mt5.symbol_info(sym)
                                exec_digits = sym_info_exec.digits if sym_info_exec else 5
                                
                                if is_long:
                                    sl_target = round(current_price - sl_dist, exec_digits)
                                    tp_target = round(current_price + (atr * tp_mult_exec), exec_digits)
                                else:
                                    sl_target = round(current_price + sl_dist, exec_digits)
                                    tp_target = round(current_price - (atr * tp_mult_exec), exec_digits)
                                
                                sl_pips = round(sl_dist / pip_size_live, 1)
                                tp_pips = round(abs(tp_target - current_price) / pip_size_live, 1)
                                
                                # Use default AA grade (3) for risk calculation
                                default_grade = 3
                                actual_risk = self.map_llm_risk_to_actual(default_grade, risk_mode)
                                verified_lots = self.calculate_lots_for_risk(
                                    sym, balance, sl_target, is_long,
                                    actual_risk, fixed_lot, risk_mode
                                )
                                
                                logger.info({
                                    "type": "trade_action",
                                    "action": f"DIRECT {action.upper()}: {pat} [No LLM | {actual_risk:.2f}% ({risk_mode}) | {verified_lots} Lots]",
                                    "tool_args": {
                                        "symbol": sym,
                                        "lots": f"SL: {sl_target} ({sl_pips} pips) | TP: {tp_target} ({tp_pips} pips) | WR: {stats.get('win_rate', 0):.1%} PF: {stats.get('profit_factor', 0):.2f}"
                                    }
                                })
                                
                                # Emit signal to LLM Signals tab
                                import datetime as _dt
                                self.signal_update.emit([{
                                    "time": _dt.datetime.now().strftime("%H:%M:%S"),
                                    "symbol": sym, "pattern": pat,
                                    "direction": action.upper(),
                                    "entry": current_price,
                                    "sl": sl_target, "tp": tp_target,
                                    "risk_pct": actual_risk,
                                    "verdict": f"{action.upper()} (Direct)" + (" [signal only]" if signal_only else "")
                                }])
                                
                                executed_symbols.add(sym)
                                if not signal_only:
                                    result = self.executor.execute_market_order(
                                        symbol=sym,
                                        is_long=is_long,
                                        volume=verified_lots,
                                        stop_loss=sl_target,
                                        take_profit=tp_target,
                                        magic_number=magic_number,
                                        comment=trade_comment
                                    )
                                    logger.info({"type": "observation", "message": f"Execution Status: {result.get('status')}"})
                                    if result.get("status") == "success":
                                        logger.info({
                                            "type": "trade_action",
                                            "action": f"EXECUTED: {pat}",
                                            "tool_args": {
                                                "symbol": sym,
                                                "lots": f"Ticket: #{result['ticket']} | Vol: {result['volume']}"
                                            }
                                        })
                                    else:
                                        logger.error(f"MT5 order failed for {pat} on {sym}: {result.get('message')}")
                                else:
                                    logger.info({"type": "observation", "message": f"[SIGNAL ONLY] {action.upper()} {pat} on {sym} — {verified_lots} lots | No execution."})
                        
                        # ══════════════════════════════════════════════════════
                        # PATH B: LLM Filter ON → Send to LLM for evaluation
                        # ══════════════════════════════════════════════════════
                        elif llm_filter:
                            enabled_model = session_config.get("enabled_model")
                            api_key = None
                            if enabled_model:
                                model_id = enabled_model.get("model_id")
                                creds = self.config_manager.model_credentials.get(model_id, {})
                                api_key = creds.get("api_key") or os.getenv('TEMP_LLM_API_KEY')
                            
                            if not enabled_model or not api_key:
                                logger.warning(f"LLM Filter ON but no model/API key provided. Bypassing Agent evaluation for {len(current_batch)} setups.")
                                for setup in current_batch:
                                    logger.info({"type": "trade_action", "action": f"Pattern Found (No LLM): {setup['pattern']} on {setup['symbol']}", "tool_args": {"symbol": setup['symbol'], "lots": f"WR: {setup['stats']['win_rate']:.1%} PF: {setup['stats']['profit_factor']:.2f}"}})
                            else:
                                model_id = enabled_model.get("model_id")
                                agent = LLMRiskAgent(model_id=model_id, api_key=api_key)
                            
                                msg_parts = []
                                if open_positions:
                                    msg_parts.append(f"{len(open_positions)} open position(s)")
                                if current_batch:
                                    msg_parts.append(f"{len(current_batch)} new setup(s)")
                                logger.info({"type": "observation", "message": f"Dispatching to LLM Agent ({model_id}): {' + '.join(msg_parts)}"})
                            
                                # Build risk context for LLM
                                mode_range = self.RISK_MODES.get(risk_mode)
                                risk_range_str = f"{mode_range[0]}%-{mode_range[1]}%" if mode_range else "fixed lots"
                                account_balance = 0
                                try:
                                    import MetaTrader5 as mt5
                                    acct = mt5.account_info()
                                    if acct:
                                        account_balance = acct.balance
                                except Exception:
                                    pass
                                # Signal-only: force virtual balance so LLM never rejects on $0
                                if signal_only and account_balance == 0:
                                    account_balance = 10000.0
                                risk_context = {
                                    "balance": account_balance,
                                    "risk_mode": risk_mode,
                                    "risk_range": risk_range_str,
                                    "portfolio_cap": self.PORTFOLIO_CAPS.get(risk_mode, 10.0),
                                    "trade_start_hour": trade_start_hour,
                                    "trade_end_hour": trade_end_hour,
                                    "current_broker_hour": now.hour,
                                    "signal_only": signal_only
                                }
                            
                                decisions = agent.evaluate_unified(current_batch, open_positions, risk_context)
                            
                                if not decisions:
                                    logger.warning("LLM returned no valid decisions for the batch.")
                            
                                executed_symbols = set()  # 1 trade per pair per bar
                                for i, decision in enumerate(decisions):
                                    decision_type = decision.get("type", "entry")
                                    sym = decision.get("symbol", "UNKNOWN")
                                    action = decision.get("action", "pass").lower()
                                
                                    # --- Handle Position Monitoring Decisions ---
                                    if decision_type == "monitor":
                                        ticket = decision.get("ticket", 0)
                                        if action == "exit":
                                            logger.info({"type": "trade_action", "action": f"LLM EXIT: {sym} (Ticket #{ticket})", "tool_args": {"symbol": sym, "lots": f"INVALIDATION — {decision.get('justification', 'No reason')}"}})
                                            closed = self.executor.close_all_positions(sym)
                                            logger.info({"type": "observation", "message": f"Emergency Exit. Positions closed: {closed}"})
                                        else:
                                            logger.info({"type": "observation", "message": f"Position HOLD: {sym} #{ticket} — {decision.get('justification', 'OK')}"})
                                        continue
                                
                                    # --- Handle Entry Decisions ---
                                    # Match setup by symbol+pattern (not by index — monitor decisions shift it)
                                    if "symbol" not in decision:
                                        decision["symbol"] = sym
                                    if "pattern" not in decision:
                                        # Try to find matching setup
                                        for s in current_batch:
                                            if s.get('symbol') == sym:
                                                decision["pattern"] = s.get("pattern", "UNKNOWN")
                                                break
                                
                                    pat = decision.get("pattern", "UNKNOWN")
                                
                                    # Find matching setup data
                                    setup_data = None
                                    for s in current_batch:
                                        if s.get('symbol') == sym and s.get('pattern') == pat:
                                            setup_data = s
                                            break
                                    if setup_data is None:
                                        for s in current_batch:
                                            if s.get('symbol') == sym:
                                                setup_data = s
                                                break
                                
                                    if action == "close":
                                        logger.info({"type": "trade_action", "action": f"LLM FORCE CLOSE: {sym}", "tool_args": {"symbol": sym, "lots": "Closing all positions."}})
                                        closed = self.executor.close_all_positions(sym)
                                        logger.info({"type": "observation", "message": f"Emergency Close Executed. Positions closed: {closed}"})
                                    
                                    elif action in ["buy", "sell"]:
                                        # SAFETY: Reject if LLM inverted the signal direction
                                        if setup_data:
                                            orig_sig = setup_data.get('signal_direction', 0)
                                            if orig_sig > 0 and action == "sell":
                                                logger.warning({"type": "observation", "message": f"⚠️ REJECTED {pat} on {sym}: LLM returned SELL but pattern signal is BUY. Direction inversion blocked."})
                                                continue
                                            if orig_sig < 0 and action == "buy":
                                                logger.warning({"type": "observation", "message": f"⚠️ REJECTED {pat} on {sym}: LLM returned BUY but pattern signal is SELL. Direction inversion blocked."})
                                                continue
                                    
                                        # 1 trade per pair per bar
                                        if sym in executed_symbols:
                                            logger.info({"type": "observation", "message": f"SKIPPED {pat} on {sym}: already traded this pair this bar."})
                                            continue
                                    
                                        is_exec_long = (action == "buy")
                                        # SAFETY CLAMP: Adapt LLM values to safe bounds, never reject
                                        sl_mult = max(2.0, min(float(decision.get('stop_loss_atr_multiplier', 2.0)), 5.0))
                                        tp_mult = max(2.0, min(float(decision.get('take_profit_atr_multiplier', 2.0)), 5.0))
                                    
                                        # Parse grade (A to AAA+) → numeric (1-6)
                                        grade_str = decision.get('grade', 'AA').upper().strip()
                                        grade_value = self.GRADE_MAP.get(grade_str, 3)  # Default AA if unknown
                                    
                                        if not setup_data:
                                            logger.error(f"Cannot execute {pat} on {sym}: No matching setup data found.")
                                            continue
                                        atr = setup_data['risk_params']['atr_value']
                                        balance = setup_data['risk_params']['account_balance']
                                    
                                        import MetaTrader5 as mt5
                                        tick = mt5.symbol_info_tick(sym)
                                        if not tick:
                                            logger.error(f"Cannot execute {pat} on {sym}: No tick data.")
                                            continue
                                        current_price = tick.ask if is_exec_long else tick.bid
                                    
                                        sl_dist = atr * sl_mult
                                        # Enforce minimum 10-pip SL
                                        pip_size_live, _ = MT5DataFetcher.pip_info(sym)
                                        if sl_dist < pip_size_live * 10:
                                            sl_dist = pip_size_live * 10
                                    
                                        # Get proper decimal precision from MT5
                                        sym_info_exec = mt5.symbol_info(sym)
                                        exec_digits = sym_info_exec.digits if sym_info_exec else 5
                                    
                                        if is_exec_long:
                                            sl_target = round(current_price - sl_dist, exec_digits)
                                            tp_target = round(current_price + (atr * tp_mult), exec_digits)
                                        else:
                                            sl_target = round(current_price + sl_dist, exec_digits)
                                            tp_target = round(current_price - (atr * tp_mult), exec_digits)
                                    
                                        # Calculate distances in pips for readable logging
                                        sl_pips = round(sl_dist / pip_size_live, 1)
                                        tp_pips = round(abs(tp_target - current_price) / pip_size_live, 1)
                                        
                                        # Map grade to actual risk % via risk mode
                                        actual_risk = self.map_llm_risk_to_actual(grade_value, risk_mode)
                                        verified_lots = self.calculate_lots_for_risk(
                                            sym, balance, sl_target, is_exec_long,
                                            actual_risk, fixed_lot, risk_mode
                                        )
                                    
                                        logger.info({
                                            "type": "trade_action", 
                                            "action": f"LLM EXECUTING {action.upper()}: {pat} [Grade {grade_str} → {actual_risk:.2f}% ({risk_mode}) | {verified_lots} Lots]", 
                                            "tool_args": {
                                                "symbol": sym, 
                                                "lots": f"SL: {sl_target} ({sl_pips} pips, x{sl_mult} ATR) | TP: {tp_target} ({tp_pips} pips, x{tp_mult} ATR)"
                                            }
                                        })
                                        logger.info({"type": "final_answer", "text": f"**Justification [{sym}]:** {decision.get('justification')}"})
                                    
                                        # Emit approved signal to LLM Signals tab
                                        import datetime as _dt
                                        self.signal_update.emit([{
                                            "time": _dt.datetime.now().strftime("%H:%M:%S"),
                                            "symbol": sym, "pattern": pat,
                                            "direction": action.upper(),
                                            "entry": current_price,
                                            "sl": sl_target, "tp": tp_target,
                                            "risk_pct": actual_risk,
                                            "verdict": f"{action.upper()} ({grade_str})" + (" [signal only]" if signal_only else "")
                                        }])
                                    
                                        executed_symbols.add(sym)  # Mark pair as traded this bar
                                        if not signal_only:
                                            result = self.executor.execute_market_order(
                                                symbol=sym,
                                                is_long=is_exec_long,
                                                volume=verified_lots,
                                                stop_loss=sl_target,
                                                take_profit=tp_target,
                                                magic_number=magic_number,
                                                comment=trade_comment
                                            )
                                            logger.info({"type": "observation", "message": f"Execution Status: {result.get('status')}"})
                                        
                                            if result.get("status") == "success":
                                                logger.info({
                                                    "type": "trade_action",
                                                    "action": f"EXECUTED: {pat}",
                                                    "tool_args": {
                                                        "symbol": sym,
                                                        "lots": f"Ticket: #{result['ticket']} | Vol: {result['volume']}"
                                                    }
                                                })
                                            else:
                                                logger.error(f"MT5 order failed for {pat} on {sym}: {result.get('message')}")
                                        else:
                                            logger.info({"type": "observation", "message": f"[SIGNAL ONLY] {action.upper()} {pat} on {sym} — {verified_lots} lots | No execution."})
                                    
                                    else:
                                        logger.warning(f"LLM DENIED {pat} on {sym}: {decision.get('justification', 'No justification')}")
                                        import datetime as _dt
                                        # Look up matching setup by symbol+pattern (not by index — monitor decisions shift it)
                                        setup_data = None
                                        for s in current_batch:
                                            if s.get('symbol') == sym and s.get('pattern') == pat:
                                                setup_data = s
                                                break
                                        if setup_data is None and current_batch:
                                            # Fallback: try by symbol only
                                            for s in current_batch:
                                                if s.get('symbol') == sym:
                                                    setup_data = s
                                                    break
                                    
                                        sig_dir = setup_data.get('signal_direction', 0) if setup_data else 0
                                        orig_dir = "BUY" if sig_dir > 0 else "SELL" if sig_dir < 0 else "?"
                                    
                                        # Get entry/SL/TP from setup data for display
                                        risk_p = setup_data.get('risk_params', {}) if setup_data else {}
                                        ctx_p = setup_data.get('market_context', {}) if setup_data else {}
                                    
                                        self.signal_update.emit([{
                                            "time": _dt.datetime.now().strftime("%H:%M:%S"),
                                            "symbol": sym, "pattern": pat,
                                            "direction": orig_dir,
                                            "entry": ctx_p.get('current_price', 0),
                                            "sl": risk_p.get('stop_loss', 0),
                                            "tp": risk_p.get('tp_1_5_rr', 0),
                                            "risk_pct": 0,
                                            "verdict": f"{orig_dir} - Rejected"
                                        }])
                                    
                    current_batch.clear()
                
                first_run = False
                self._validated_cache = validated_cache  # Expose for GUI save
                self.console_ready.emit()  # Signal: patterns loaded, console can hide
                self.training_complete.emit(validated_cache)  # Enable Save button
                
                # ═══════════════════════════════════════════════════════════════
                # CATCH-UP PHASE: Retroactively evaluate candles missed during training
                # ═══════════════════════════════════════════════════════════════
                if training_started_at is not None:
                    import datetime as _dt_catchup
                    from news_filter import get_broker_time as _get_bt_catchup
                    catchup_now = _get_bt_catchup()
                    
                    logger.info({"type": "observation", "message": f"Training took from {training_started_at.strftime('%H:%M')} to {catchup_now.strftime('%H:%M')}. Checking for missed candles..."})
                    self.status_update.emit("Catching up on candles missed during training...", "status_label_pending")
                    
                    catchup_batch = []
                    
                    for tf_name_cu, tf_cfg_cu in valid_tfs.items():
                        if not self._is_running:
                            break
                        
                        tf_minutes = tf_cfg_cu.get("minutes", 60)
                        
                        # Compute candle boundaries that closed during training
                        start_total = training_started_at.hour * 60 + training_started_at.minute
                        end_total = catchup_now.hour * 60 + catchup_now.minute
                        
                        # Handle day wrap (e.g. training started at 23:00, now it's 01:00)
                        if end_total <= start_total:
                            end_total += 24 * 60
                        
                        # Find the first candle boundary AFTER training started
                        first_boundary = ((start_total // tf_minutes) + 1) * tf_minutes
                        
                        missed_boundaries = []
                        b = first_boundary
                        while b <= end_total:
                            missed_boundaries.append(b % (24 * 60))
                            b += tf_minutes
                        
                        # The LAST boundary is the current candle close — live loop will handle it
                        if missed_boundaries:
                            missed_boundaries.pop()
                        
                        if not missed_boundaries:
                            logger.info({"type": "observation", "message": f"Catch-up [{tf_name_cu}]: No missed candles (training was fast enough)."})
                            continue
                        
                        logger.info({"type": "observation", "message": f"Catch-up [{tf_name_cu}]: {len(missed_boundaries)} candle(s) missed during training. Evaluating..."})
                        
                        for symbol_cu in symbols:
                            if not self._is_running:
                                break
                            
                            cache_key_cu = (symbol_cu, tf_name_cu)
                            cached_data_cu = validated_cache.get(cache_key_cu, {})
                            if not cached_data_cu:
                                continue
                            cached_patterns_cu = list(cached_data_cu.keys())
                            
                            # Check news blackout
                            news_blocked_cu = is_symbol_blocked(symbol_cu, buffer_before=news_before, buffer_after=news_after)
                            if news_blocked_cu:
                                continue
                            
                            # Fetch data (500 bars — all missed candles are in here)
                            tf_const_cu = tf_cfg_cu.get("tf_const", 16385)
                            df_cu = self.fetcher.fetch_symbol(symbol_cu, tf_const_cu, 500, tf_name_cu, use_cache=False)
                            if df_cu is None or df_cu.empty:
                                continue
                            
                            # Scan only validated patterns
                            signals_cu = self.registry.scan_specific(df_cu, cached_patterns_cu)
                            
                            # Time filter
                            hours_cu = df_cu.index.hour
                            outside_cu = (hours_cu < trade_start_hour) | (hours_cu >= trade_end_hour)
                            signals_cu.loc[outside_cu] = 0
                            
                            # Check each missed boundary
                            for missed_min in missed_boundaries:
                                missed_hour = missed_min // 60
                                missed_minute = missed_min % 60
                                
                                # Find the bar that corresponds to the START of this candle
                                candle_start_min = missed_min - tf_minutes
                                if candle_start_min < 0:
                                    candle_start_min += 24 * 60
                                candle_start_h = candle_start_min // 60
                                candle_start_m = candle_start_min % 60
                                
                                # Find matching bar in the DataFrame by hour and minute
                                matching_bars = df_cu[
                                    (df_cu.index.hour == candle_start_h) & 
                                    (df_cu.index.minute == candle_start_m)
                                ]
                                if matching_bars.empty:
                                    continue
                                
                                # Use the LAST matching bar (most recent day)
                                bar_idx = df_cu.index.get_loc(matching_bars.index[-1])
                                
                                for pat_cu in cached_patterns_cu:
                                    if pat_cu not in signals_cu.columns:
                                        continue
                                    
                                    sig_val = signals_cu[pat_cu].iloc[bar_idx]
                                    if sig_val == 0:
                                        continue
                                    
                                    logger.info({"type": "observation", "message": f"Catch-up SIGNAL: {pat_cu} on {symbol_cu} at {candle_start_h:02d}:{candle_start_m:02d} ({tf_name_cu})"})
                                    
                                    # Build market context from the catch-up bar
                                    is_long_cu = (sig_val > 0)
                                    current_price_cu = df_cu['close'].iloc[bar_idx]
                                    
                                    ema20_cu = utils.ema(df_cu['close'], 20).iloc[bar_idx]
                                    ema50_cu = utils.ema(df_cu['close'], 50).iloc[bar_idx]
                                    ema200_cu = utils.ema(df_cu['close'], 200).iloc[bar_idx]
                                    rsi_cu = utils.rsi(df_cu, 14).iloc[bar_idx]
                                    bb_up_cu, bb_mid_cu, bb_low_cu = utils.bollinger(df_cu, 20, 2.0)
                                    macd_line_cu, macd_signal_cu, macd_hist_cu = utils.macd(df_cu)
                                    adx_cu, plus_di_cu, minus_di_cu = utils.adx(df_cu)
                                    atr_cu = utils.atr(df_cu, 14).iloc[bar_idx]
                                    
                                    import MetaTrader5 as mt5
                                    tick_cu = mt5.symbol_info_tick(symbol_cu)
                                    sym_info_cu = mt5.symbol_info(symbol_cu)
                                    spread_cu = (tick_cu.ask - tick_cu.bid) / sym_info_cu.point if tick_cu and sym_info_cu else 0.0
                                    spread_price_cu = tick_cu.ask - tick_cu.bid if tick_cu else 0.0
                                    mid_cu = (tick_cu.ask + tick_cu.bid) / 2 if tick_cu else current_price_cu
                                    spread_pct_cu = (spread_price_cu / mid_cu) * 100 if mid_cu > 0 else 0.0
                                    sym_digits_cu = sym_info_cu.digits if sym_info_cu else 5
                                    
                                    market_ctx_cu = {
                                        "current_price": round(current_price_cu, sym_digits_cu),
                                        "trend_ema_20": round(ema20_cu, sym_digits_cu),
                                        "trend_ema_50": round(ema50_cu, sym_digits_cu),
                                        "trend_ema_200": round(ema200_cu, sym_digits_cu),
                                        "rsi_14": round(rsi_cu, 2),
                                        "bb_upper": round(bb_up_cu.iloc[bar_idx], sym_digits_cu),
                                        "bb_mid": round(bb_mid_cu.iloc[bar_idx], sym_digits_cu),
                                        "bb_lower": round(bb_low_cu.iloc[bar_idx], sym_digits_cu),
                                        "macd_line": round(macd_line_cu.iloc[bar_idx], sym_digits_cu + 1),
                                        "macd_signal": round(macd_signal_cu.iloc[bar_idx], sym_digits_cu + 1),
                                        "macd_histogram": round(macd_hist_cu.iloc[bar_idx], sym_digits_cu + 1),
                                        "adx_14": round(adx_cu.iloc[bar_idx], 2),
                                        "plus_di": round(plus_di_cu.iloc[bar_idx], 2),
                                        "minus_di": round(minus_di_cu.iloc[bar_idx], 2),
                                        "atr_14": round(atr_cu, sym_digits_cu),
                                        "spread_points": round(spread_cu, 1),
                                        "spread_percentage": round(spread_pct_cu, 4)
                                    }
                                    
                                    risk_params_cu = self.calculate_atr_risk_metrics(df_cu, symbol_cu, current_price_cu, is_long_cu)
                                    if signal_only and risk_params_cu.get('account_balance', 0) == 0:
                                        risk_params_cu['account_balance'] = 10000.0
                                    
                                    tier_lots_cu = self.calculate_tier_lots(symbol_cu, risk_params_cu['account_balance'], risk_params_cu['stop_loss'], is_long_cu, risk_mode, fixed_lot)
                                    risk_params_cu['tier_1_lots'] = tier_lots_cu.get("1%")
                                    risk_params_cu['tier_2_lots'] = tier_lots_cu.get("2%")
                                    risk_params_cu['tier_3_lots'] = tier_lots_cu.get("3%")
                                    
                                    stats_cu = cached_data_cu.get(pat_cu, {"win_rate": 0, "profit_factor": 0, "trade_count": 0, "avg_gain": 0})
                                    htf_ctx_cu = self.get_htf_context(symbol_cu, tf_name_cu)
                                    
                                    catchup_batch.append({
                                        "symbol": symbol_cu,
                                        "timeframe": tf_name_cu,
                                        "pattern": pat_cu,
                                        "signal_direction": int(sig_val),
                                        "stats": stats_cu,
                                        "risk_params": risk_params_cu,
                                        "market_context": market_ctx_cu,
                                        "htf_context": htf_ctx_cu
                                    })
                    
                    # Dispatch catch-up batch to LLM (reuse existing LLM evaluation)
                    if catchup_batch and self._is_running:
                        logger.info({"type": "observation", "message": f"Catch-up: {len(catchup_batch)} signal(s) found on missed candles. Dispatching to LLM..."})
                        
                        enabled_model = session_config.get("enabled_model")
                        api_key_cu = None
                        if enabled_model:
                            model_id_cu = enabled_model.get("model_id")
                            creds_cu = self.config_manager.model_credentials.get(model_id_cu, {})
                            api_key_cu = creds_cu.get("api_key") or os.getenv('TEMP_LLM_API_KEY')
                        
                        if not enabled_model or not api_key_cu:
                            for setup_cu in catchup_batch:
                                logger.info({"type": "trade_action", "action": f"Catch-up Signal (No LLM): {setup_cu['pattern']} on {setup_cu['symbol']}", "tool_args": {"symbol": setup_cu['symbol'], "lots": f"WR: {setup_cu['stats'].get('win_rate', 0):.1%} PF: {setup_cu['stats'].get('profit_factor', 0):.2f}"}})
                        else:
                            model_id_cu = enabled_model.get("model_id")
                            agent_cu = LLMRiskAgent(model_id=model_id_cu, api_key=api_key_cu)
                            
                            import MetaTrader5 as mt5
                            acct_cu = mt5.account_info()
                            balance_cu = acct_cu.balance if acct_cu else 0.0
                            if signal_only and balance_cu == 0:
                                balance_cu = 10000.0
                            
                            mode_range_cu = self.RISK_MODES.get(risk_mode)
                            risk_range_cu = f"{mode_range_cu[0]}%-{mode_range_cu[1]}%" if mode_range_cu else "fixed lots"
                            risk_ctx_cu = {
                                "balance": balance_cu,
                                "risk_mode": risk_mode,
                                "risk_range": risk_range_cu,
                                "portfolio_cap": self.PORTFOLIO_CAPS.get(risk_mode, 10.0),
                                "trade_start_hour": trade_start_hour,
                                "trade_end_hour": trade_end_hour,
                                "current_broker_hour": catchup_now.hour,
                                "signal_only": signal_only
                            }
                            
                            decisions_cu = agent_cu.evaluate_unified(catchup_batch, [], risk_ctx_cu)
                            
                            if decisions_cu:
                                for dec_cu in decisions_cu:
                                    sym_cu = dec_cu.get("symbol", "UNKNOWN")
                                    pat_name_cu = dec_cu.get("pattern", "UNKNOWN")
                                    action_cu = dec_cu.get("action", "pass").lower()
                                    
                                    if action_cu in ["buy", "sell"]:
                                        # Find matching setup
                                        su_data = None
                                        for s in catchup_batch:
                                            if s.get('symbol') == sym_cu and s.get('pattern') == pat_name_cu:
                                                su_data = s
                                                break
                                        if su_data is None:
                                            continue
                                        
                                        # Direction inversion guard
                                        orig_sig_cu = su_data.get('signal_direction', 0)
                                        if (orig_sig_cu > 0 and action_cu == "sell") or (orig_sig_cu < 0 and action_cu == "buy"):
                                            logger.warning({"type": "observation", "message": f"⚠️ Catch-up REJECTED {pat_name_cu} on {sym_cu}: direction inversion."})
                                            continue
                                        
                                        is_exec_long_cu = (action_cu == "buy")
                                        sl_mult_cu = max(2.0, min(float(dec_cu.get('stop_loss_atr_multiplier', 2.0)), 5.0))
                                        tp_mult_cu = max(2.0, min(float(dec_cu.get('take_profit_atr_multiplier', 2.0)), 5.0))
                                        grade_str_cu = dec_cu.get('grade', 'AA').upper().strip()
                                        grade_value_cu = self.GRADE_MAP.get(grade_str_cu, 3)
                                        
                                        atr_cu_val = su_data['risk_params']['atr_value']
                                        balance_exec_cu = su_data['risk_params']['account_balance']
                                        
                                        tick_exec_cu = mt5.symbol_info_tick(sym_cu)
                                        if not tick_exec_cu:
                                            continue
                                        price_exec_cu = tick_exec_cu.ask if is_exec_long_cu else tick_exec_cu.bid
                                        
                                        sl_dist_cu = atr_cu_val * sl_mult_cu
                                        pip_size_cu, _ = MT5DataFetcher.pip_info(sym_cu)
                                        if sl_dist_cu < pip_size_cu * 10:
                                            sl_dist_cu = pip_size_cu * 10
                                        
                                        sym_info_exec_cu = mt5.symbol_info(sym_cu)
                                        digits_cu = sym_info_exec_cu.digits if sym_info_exec_cu else 5
                                        
                                        if is_exec_long_cu:
                                            sl_price_cu = round(price_exec_cu - sl_dist_cu, digits_cu)
                                            tp_price_cu = round(price_exec_cu + (atr_cu_val * tp_mult_cu), digits_cu)
                                        else:
                                            sl_price_cu = round(price_exec_cu + sl_dist_cu, digits_cu)
                                            tp_price_cu = round(price_exec_cu - (atr_cu_val * tp_mult_cu), digits_cu)
                                        
                                        sl_pips_cu = round(sl_dist_cu / pip_size_cu, 1)
                                        tp_pips_cu = round(abs(tp_price_cu - price_exec_cu) / pip_size_cu, 1)
                                        
                                        actual_risk_cu = self.map_llm_risk_to_actual(grade_value_cu, risk_mode)
                                        lots_cu = self.calculate_lots_for_risk(sym_cu, balance_exec_cu, sl_price_cu, is_exec_long_cu, actual_risk_cu, fixed_lot, risk_mode)
                                        
                                        logger.info({
                                            "type": "trade_action",
                                            "action": f"CATCH-UP LLM EXECUTING {action_cu.upper()}: {pat_name_cu} [Grade {grade_str_cu} → {actual_risk_cu:.2f}% ({risk_mode}) | {lots_cu} Lots]",
                                            "tool_args": {"symbol": sym_cu, "lots": f"SL: {sl_price_cu} ({sl_pips_cu} pips) | TP: {tp_price_cu} ({tp_pips_cu} pips)"}
                                        })
                                        logger.info({"type": "final_answer", "text": f"**Catch-up Justification [{sym_cu}]:** {dec_cu.get('justification')}"})
                                        
                                        # Emit to LLM Signals tab
                                        import datetime as _dt_cu
                                        self.signal_update.emit([{
                                            "time": _dt_cu.datetime.now().strftime("%H:%M:%S"),
                                            "symbol": sym_cu, "pattern": pat_name_cu,
                                            "direction": action_cu.upper(),
                                            "entry": price_exec_cu,
                                            "sl": sl_price_cu, "tp": tp_price_cu,
                                            "risk_pct": actual_risk_cu,
                                            "verdict": f"CATCH-UP {action_cu.upper()} ({grade_str_cu})" + (" [signal only]" if signal_only else "")
                                        }])
                                        
                                        if not signal_only:
                                            result_cu = self.executor.execute_market_order(
                                                symbol=sym_cu, is_long=is_exec_long_cu,
                                                volume=lots_cu, stop_loss=sl_price_cu,
                                                take_profit=tp_price_cu, magic_number=magic_number,
                                                comment=trade_comment
                                            )
                                            logger.info({"type": "observation", "message": f"Catch-up Execution: {result_cu.get('status')}"})
                                        else:
                                            logger.info({"type": "observation", "message": f"[SIGNAL ONLY] Catch-up {action_cu.upper()} {pat_name_cu} on {sym_cu} — no execution."})
                                    else:
                                        logger.warning(f"Catch-up LLM DENIED {pat_name_cu} on {sym_cu}: {dec_cu.get('justification', 'No justification')}")
                    else:
                        logger.info({"type": "observation", "message": "Catch-up: No signals found on missed candles."})
                    
                    training_started_at = None  # Only run catch-up once
                # ═══════════════════════════════════════════════════════════════
                
                # CRITICAL: After first_run backtest, wait for the current candle to close
                # before starting live trading. Never trade on an incomplete candle.
                logger.info({"type": "observation", "message": "Backtest complete. Waiting for current candle to close before going live..."})
                self.status_update.emit("Backtest done. Waiting for candle close...", "status_label_pending")
            
            if all_session_trades:
                all_session_trades.sort(key=lambda t: t.exit_time)
                closed_trades = [t for t in all_session_trades if t.exit_time is not None]
                if closed_trades:
                    # Use a clean 0 to N index for X-axis to avoid PyQtGraph float-timestamp overflow issues
                    exits = list(range(len(closed_trades)))
                    pnls = [t.pnl_pips for t in closed_trades]
                    import numpy as np
                    cum = np.cumsum(pnls).tolist()
                    self.equity_update.emit(exits, cum)

            # Moved out of the loop
            self.status_update.emit("Engine Terminated", "status_label_stopped")
            
        except Exception as e:
            logger.error(f"Trading Engine Error: {e}", exc_info=True)
            self.status_update.emit("Error", "status_label_error")
            
        finally:
            self.fetcher.shutdown()
            self._finish()

    def stop(self):
        """Called when 'STOP' is clicked."""
        self._is_running = False
        self.status_update.emit("Stopping...", "status_label_pending")

    def _finish(self):
        self._is_running = False
        self.connection_status.emit(False)
        self.finished.emit()
