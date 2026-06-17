"""
News Calendar Filter
Loads news events from the local calendar file on startup.
Blocks NEW ENTRIES on affected currency pairs within a configurable buffer window.
USD news = TOTAL BLACKOUT on ALL pairs (entries only — exits/holds NOT blocked).

IMPORTANT: Times in news calendar must match BROKER SERVER TIME.
We fetch broker time directly from MT5 — no UTC translation.
"""

import os
import csv
import logging
from datetime import datetime, timedelta
from typing import Set

logger = logging.getLogger(__name__)

_CALENDAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news_calendar.csv")

# Default buffer: 2 hours before and after
_BUFFER_BEFORE_MINUTES = 120
_BUFFER_AFTER_MINUTES = 120

# Module-level cache of events (fetched on startup)
_cached_events = []
_news_loaded = False


def load_news_calendar() -> bool:
    """Load news events from the local calendar file.
    Returns True if events were loaded, False otherwise.
    """
    if not os.path.isfile(_CALENDAR_PATH):
        logger.error(f"[NEWS] Local news calendar not found: {_CALENDAR_PATH}")
        return False
    _load_events_from_csv()
    return _news_loaded


def _load_events_from_csv():
    """Load events from local CSV backup."""
    global _cached_events, _news_loaded
    
    try:
        events = []
        with open(_CALENDAR_PATH, 'r', encoding='utf-8') as f:
            lines = [line for line in f if not line.strip().startswith('#')]
        reader = csv.DictReader(lines)
        for row in reader:
            events.append({
                'datetime': row.get('datetime', '').strip(),
                'currency': row.get('currency', '').strip().upper(),
                'event': row.get('event', '').strip(),
            })
        _cached_events = events
        _news_loaded = True
        logger.info(f"[NEWS] Loaded {len(events)} events from local CSV backup.")
    except Exception as e:
        logger.warning(f"[NEWS] Could not read local CSV: {e}")


def is_news_loaded() -> bool:
    """Check if news data has been loaded."""
    return _news_loaded


def get_broker_time() -> datetime:
    """Get current broker server time from MT5. Falls back to local time if MT5 unavailable."""
    try:
        import MetaTrader5 as mt5
        # Get last tick time from any popular symbol
        for sym in ["EURUSD", "GBPUSD", "USDJPY"]:
            tick = mt5.symbol_info_tick(sym)
            if tick is not None:
                return datetime.utcfromtimestamp(tick.time)
    except Exception:
        pass
    # Fallback: local time (should rarely happen since MT5 is always connected during runs)
    return datetime.now()


def get_blocked_currencies(now: datetime = None, buffer_before: int = None, buffer_after: int = None) -> Set[str]:
    """
    Returns set of currency codes (e.g. {'USD', 'GBP'}) that are currently
    inside a news blackout window.
    Uses BROKER TIME directly — no UTC conversion.
    
    buffer_before: minutes before the event to start blocking (default: 120)
    buffer_after: minutes after the event to stop blocking (default: 120)
    """
    if now is None:
        now = get_broker_time()
    
    if buffer_before is None:
        buffer_before = _BUFFER_BEFORE_MINUTES
    if buffer_after is None:
        buffer_after = _BUFFER_AFTER_MINUTES
    
    blocked = set()
    
    for evt in _cached_events:
        dt_str = evt.get('datetime', '').strip().replace('T', ' ')
        currency = evt.get('currency', '').strip().upper()
        
        if not dt_str or not currency:
            continue
        
        try:
            # Handle both "2026-02-25 14:30" and "2026-02-25T14:30" formats
            if len(dt_str) == 16:  # "2026-02-25 14:30"
                event_time = datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
            else:
                event_time = datetime.strptime(dt_str[:16], '%Y-%m-%d %H:%M')
        except ValueError:
            continue
        
        window_start = event_time - timedelta(minutes=buffer_before)
        window_end = event_time + timedelta(minutes=buffer_after)
        
        if window_start <= now <= window_end:
            blocked.add(currency)
    
    return blocked


def is_symbol_blocked(symbol: str, now: datetime = None, buffer_before: int = None, buffer_after: int = None) -> bool:
    """Check if a symbol is blocked for NEW ENTRIES due to news.
    USD news = TOTAL BLACKOUT on ALL pairs (market-wide impact).
    Other currencies only block pairs containing that currency.
    NOTE: This only blocks entries — open positions can still be managed (hold/exit).
    """
    blocked = get_blocked_currencies(now, buffer_before, buffer_after)
    if not blocked:
        return False
    # USD news -> block everything
    if 'USD' in blocked:
        return True
    symbol_upper = symbol.upper()
    for ccy in blocked:
        if ccy in symbol_upper:
            return True
    return False


def get_blocked_symbols(symbols: list, now: datetime = None, buffer_before: int = None, buffer_after: int = None) -> dict:
    """
    Given a list of symbols, returns {symbol: [list of blocking currencies]} 
    for all blocked symbols. USD news blocks ALL symbols.
    """
    blocked_ccys = get_blocked_currencies(now, buffer_before, buffer_after)
    if not blocked_ccys:
        return {}
    
    has_usd = 'USD' in blocked_ccys
    result = {}
    for sym in symbols:
        if has_usd:
            result[sym] = ['USD']
        else:
            sym_upper = sym.upper()
            matching = [c for c in blocked_ccys if c in sym_upper]
            if matching:
                result[sym] = matching
    return result
